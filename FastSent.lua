local FastSent = torch.class('sentenceembedding.FastSent')

function FastSent:__init(config)
  self.name               = 'FastSent'
  self.init_learning_rate = config.learning_rate or 0.0001
  self.learning_rate = config.learning_rate or 0.0001
  self.batch_size    = config.batch_size    or 10
  self.reg           = config.reg           or 0
  self.grad_clip     = config.grad_clip     or 10
  self.current_epoch = 1

  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- optimizer configuration
  self.optim_state =
  { learningRate = self.learning_rate,
    weightDecay = 1e-5
  }

  self.output_progress = config.output_progress or false
  self.progress_writer = config.progress_writer

  -- Set Objective as minimize Negative Log Likelihood
  -- Remember to set the size_average to false to use the effect of weight!!
  -- self.criterion = nn.ClassNLLCriterion(self.class_weight, false)
  self.criterion = nn.ClassNLLCriterion()

  -- initialize SkipThought model
  -- Input model mapping the word index to embedding vectors
  self.input_module = nn.Sequential()
  local lookup = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  lookup.weight:copy(config.emb_vecs)
  self.input_module:add(lookup)

  -- initialize Probability model
  self.prob_module = nn.Sequential()
    :add(nn.Linear(self.emb_dim, self.emb_vecs:size(1)))
    :add(nn.LogSoftMax())

  -- For getting all the parameters for the SkipThought model
  local modules = nn.Parallel()
    :add(self.input_module)
    :add(self.prob_module)

  self.params, self.grad_params = modules:getParameters()
end

function FastSent:train(dataset, corpus)
  self.input_module:training()
  self.prob_module:training()

  local indices = torch.randperm(dataset.size)
  local train_loss = 0
  local data_count = 0
  local average_progress_train_loss = 0
  local average_count = 0
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    -- self.learning_rate = self.init_learning_rate*math.exp(-(i + (self.current_epoch-1)*20)*0.05)
    -- -- Decay the learning rate every 500 epochs
    -- if i%100 == 0 then
    --   -- self.learning_rate = self.learning_rate*0.9
    --   print('Current learning rate:', self.learning_rate)
    --   self.progress_writer:write_string(
    --     string.format('\t%s: %.6e\n',   'Current Learning Rate', self.learning_rate))
    --   -- Stop training is the learning rate is decayed to zero
    --   if self.learning_rate == 0 then
    --     break
    --   end
    -- end

    -- if i%5 == 1 then
    --   if average_count>0 then
    --     average_progress_train_loss = average_progress_train_loss/average_count
    --     print('Training loss:', average_progress_train_loss)
    --     self.progress_writer:write_string(
    --       string.format('%s: %.4f\n',   'Training Loss', average_progress_train_loss))
    --     average_progress_train_loss = 0
    --     average_count = 0
    --   end
    -- end

    local feval = function(x)
      if x ~= self.params then
        self.params:copy(x)
      end
      self.grad_params:zero()

      local batch_loss = 0
      -- For each datapoint in current batch
      for j = 1, batch_size do
        local idx = indices[i + j - 1]

        -- Initialze each sentence with its token mapped to embedding vectors
        local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx =
          self:load_input_sentences(idx, dataset, corpus)
        if embedding_sentence_with_vocab_idx == nil or
          pre_sentence_with_vocab_idx == nil or
          post_sentence_with_vocab_idx == nil then
            print('Sentence Loading error for index:')
            print(idx)
            goto continue
        end

        local embedding_sentence = self.input_module:forward(embedding_sentence_with_vocab_idx)

        if embedding_sentence == nil then
          print('Sentence too short.')
          goto continue
        end

        local pre_target, post_target
        pre_target = pre_sentence_with_vocab_idx:sub(1, -2)
        post_target = post_sentence_with_vocab_idx:sub(1, -2)
        local target = torch.cat(pre_target, post_target, 1)

        local embedding_sentence_vector = torch.sum(embedding_sentence, 1)
        local pro_module_input = torch.repeatTensor(embedding_sentence_vector,target:size(1),1)

        local prediction_output = self.prob_module:forward(pro_module_input)

        -- Create the prediction target from the pre and post sentences
        local pre_target, post_target
        pre_target = pre_sentence_with_vocab_idx:sub(1, -2)
        post_target = post_sentence_with_vocab_idx:sub(1, -2)
        local target = torch.cat(pre_target, post_target, 1)

        local sentence_loss = self.criterion:forward(prediction_output, target)
        batch_loss = batch_loss + sentence_loss
        data_count = data_count + 1
        -- print(sentence_loss)
        -- print('sentence_loss:', sentence_loss)
        -- average_progress_train_loss = average_progress_train_loss + sentence_loss
        -- average_count = average_count + 1

        -- Starting the backward process
        local sentence_grad = self.criterion:backward(prediction_output, target)
        local prob_grad = self.prob_module:backward(pro_module_input, sentence_grad)
        -- print(prob_grad)
        local input_grad = self.input_module:backward(embedding_sentence_with_vocab_idx, prob_grad[1])

        ::continue::
      end -- Finished
      train_loss = train_loss + batch_loss
      print('Batch loss:', batch_loss/batch_size)
      self.progress_writer:write_string(
        string.format('%s: %.4f\n',   'Batch Loss', batch_loss/batch_size))

      self.grad_params:div(batch_size)


      -- regularization
      batch_loss = batch_loss + 0.5 * self.reg * self.params:norm() ^ 2 * batch_size/dataset.size
      -- Final derivatives to return after regularization:
      -- self.grad_params + self.reg*self.params
      self.grad_params:add(self.reg, self.params)
      return batch_loss, self.grad_params
    end
    -- optim.adam(feval, self.params, self.optim_state)
    optim.rmsprop(feval, self.params, self.optim_state)
  end

  train_loss = train_loss/data_count
  xlua.progress(dataset.size, dataset.size)
  -- print('Training loss', train_loss)
  self.current_epoch = self.current_epoch + 1
  return train_loss
end

function FastSent:load_input_sentences(idx, dataset, corpus)
  local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx
  -- load sentence tuple for the current training data point from the corpus
  if corpus.ids[dataset.embedding_sentence[idx]]~= nil then
    embedding_sentence_with_vocab_idx = corpus.sentences[corpus.ids[dataset.embedding_sentence[idx]]]
  else
    print('Cannot find embedding sentence for current training '..
      'data point:', dataset.embedding_sentence[idx])
    return
  end

  if corpus.ids[dataset.pre_sentence[idx]]~= nil then
    pre_sentence_with_vocab_idx =
      corpus.sentences[corpus.ids[dataset.pre_sentence[idx]]]
  else
    print('Cannot find the sentence before the embedding sentence for '..
      'current training data point:', dataset.pre_sentence[idx])
    return
  end

  if corpus.ids[dataset.post_sentence[idx]]~= nil then
    post_sentence_with_vocab_idx =
      corpus.sentences[corpus.ids[dataset.post_sentence[idx]]]
    else
    print('Cannot find the sentence after the embedding sentence for '..
      'current training data point:', dataset.post_sentence[idx])
    return
  end
  return embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx
end


function FastSent:calcluate_loss(dataset, corpus)
  self.input_module:evaluate()
  self.prob_module:evaluate()
  local total_loss = 0
  local data_count = 0
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local sentence_loss = self:calculate_loss_one_instance(i, dataset, corpus)
    if sentence_loss ~= -1 then
      total_loss = total_loss + sentence_loss
      data_count = data_count + 1
    end
  end
  xlua.progress(dataset.size, dataset.size)
  return total_loss/data_count
end

function FastSent:calculate_loss_one_instance(idx, dataset, corpus)
  -- Initialze each sentence with its token mapped to embedding vectors
  local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx =
    self:load_input_sentences(idx, dataset, corpus)
  if embedding_sentence_with_vocab_idx == nil or
    pre_sentence_with_vocab_idx == nil or
    post_sentence_with_vocab_idx == nil then
      print('Sentence Loading error for index:')
      print(idx)
      return -1
  end

  local embedding_sentence = self.input_module:forward(embedding_sentence_with_vocab_idx)

  if embedding_sentence == nil then
    print('Sentence too short.')
    return -1
  end

  local pre_target, post_target
  pre_target = pre_sentence_with_vocab_idx:sub(1, -2)
  post_target = post_sentence_with_vocab_idx:sub(1, -2)
  local target = torch.cat(pre_target, post_target, 1)

  local embedding_sentence_vector = torch.sum(embedding_sentence, 1)
  local pro_module_input = torch.repeatTensor(embedding_sentence_vector,target:size(1),1)

  local prediction_output = self.prob_module:forward(pro_module_input)

  -- Create the prediction target from the pre and post sentences
  local pre_target, post_target
  pre_target = pre_sentence_with_vocab_idx:sub(1, -2)
  post_target = post_sentence_with_vocab_idx:sub(1, -2)
  local target = torch.cat(pre_target, post_target, 1)

  local sentence_loss = self.criterion:forward(prediction_output, target)
  return sentence_loss
end

function FastSent:print_config()
  print('Configurations for the Skip Thoughts Model')
  local num_params = self.params:nElement()
  printf('%-25s = %.2e\n', 'initial learning rate', self.learning_rate)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %f\n',   'gradient clipping', self.grad_clip)
  printf('%-25s = %.2e\n',  'regularization strength', self.reg)
  printf('\n')
  printf('%-25s = %d\n',   'Word Embedding dim', self.emb_dim)
  printf('%-25s = %d\n',   'num params', num_params)
end


--
-- Serialization
--
function FastSent:save(path)
  local config = {
    learning_rate       = self.learning_rate,
    batch_size          = self.batch_size,
    reg                 = self.reg,
    grad_clip           = self.grad_clip,

    -- word embedding
    emb_vecs            = self.emb_vecs,
    emb_dim             = self.emb_dim
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function FastSent.load(path)
  local state = torch.load(path)
  local model = sentenceembedding.SkipThought.new(state.config)
  model.params:copy(state.params)
  return model
end

return SkipThought
