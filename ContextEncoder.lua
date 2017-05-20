local ContextEncoder = torch.class('sentenceembedding.ContextEncoder')

function ContextEncoder:__init(config)
  self.name                  = 'ContextEncoder'
  self.encoder_hidden_dim    = config.encoder_hidden_dim    or 50
  self.encoder_num_layers    = config.encoder_num_layers    or 1
  self.encoder_structure     = config.encoder_structure     or 'bigru'

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

  self.update_word_embedding = config.update_word_embedding or 0

  self.output_progress = config.output_progress or 0
  self.progress_writer = config.progress_writer

  -- Set Objective as minimize Negative Log Likelihood
  -- Remember to set the size_average to false to use the effect of weight!!
  -- self.criterion = nn.ClassNLLCriterion(self.class_weight, false)
  self.criterion = nn.MSECriterion()

  -- initialize ContextEncoder model
  -- Input model mapping the word index to embedding vectors
  self.input_module = nn.Sequential()
  local lookup = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  lookup.weight:copy(config.emb_vecs)
  self.input_module:add(lookup)


  -- initialize Encoder model
  local encoder_config = {
    hidden_dim = self.encoder_hidden_dim,
    num_layers = self.encoder_num_layers,
    emb_vecs   = self.emb_vecs,
    structure  = self.encoder_structure
  }
  self.encoder = sentenceembedding.Encoder(encoder_config)
  self.encoder_pre = sentenceembedding.Encoder(encoder_config)
  self.encoder_post = sentenceembedding.Encoder(encoder_config)


  -- initialize Decoder model
  local encoder_out_dim_real
  if string.starts(self.encoder_structure,'bi') then
    encoder_out_dim_real = 2*self.encoder_num_layers*self.encoder_hidden_dim
  else
    encoder_out_dim_real = self.encoder_num_layers*self.encoder_hidden_dim
  end

  self.transform_module = nn.Sequential()
    :add(nn.Linear(encoder_out_dim_real, encoder_out_dim_real))
    :add(nn.Sigmoid())


  -- For getting all the parameters for the ContextEncoder model
  local modules = nn.Parallel()
    :add(self.encoder)
    :add(self.transform_module)

  if self.update_word_embedding == 1 then
    modules:add(self.input_module)
  end

  self.params, self.grad_params = modules:getParameters()

  -- Get the number of parameters for encoder
  self.encoder_params = self.encoder:parameters()
  self.encoder_params_element_number = 0
  for i=1,#self.encoder_params do
    self.encoder_params_element_number =
      self.encoder_params_element_number + self.encoder_params[i]:nElement()
  end

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  share_params(self.encoder_pre, self.encoder)
  share_params(self.encoder_post, self.encoder)
end

function ContextEncoder:train(dataset, corpus)
  self.encoder:training()
  self.encoder_pre:training()
  self.encoder_post:training()

  local indices = torch.randperm(dataset.size)
  local train_loss = 0
  local data_count = 0
  local batch_count = 1
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    if  batch_count%50 == 0 then
      self.learning_rate = self.learning_rate*0.5
      print('Learning rate reduced to:', self.learning_rate)
      batch_count = 1
    end
    batch_count = batch_count + 1

    local feval = function(x)
      if x ~= self.params then
        self.params:copy(x)
      end
      self.grad_params:zero()

      local batch_loss = 0
      -- For each datapoint in current batch
      for j = 1, batch_size do
        local idx = indices[i + j - 1]

        -- load sentence tuple for the current training data point from the corpus
        local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx =
          self:load_input_sentences(idx, dataset, corpus)
        if embedding_sentence_with_vocab_idx == nil or
          pre_sentence_with_vocab_idx == nil or
          post_sentence_with_vocab_idx == nil then
            print('Sentence Loading error for index:')
            print(idx)
            goto continue
        end


        -- Initialze each sentence with its token mapped to embedding vectors
        -- Initialze each sentence with its token mapped to embedding vectors
        local embedding_sentence, pre_sentence, post_sentence =
          self:input_module_forward(embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx,
          post_sentence_with_vocab_idx)

        if embedding_sentence == nil or pre_sentence == nil or post_sentence == nil then
          print('Sentence too short.')
          return -1
        end

        -- Start the forward process
        local encoder_result = self:encoder_forward(embedding_sentence, self.encoder)
        local encoder_pre_result = self:encoder_forward(pre_sentence, self.encoder_pre)
        local encoder_post_result = self:encoder_forward(post_sentence, self.encoder_post)

        -- Concatenate encoder_result for pre and post sentence
        local transformed_encoder_result = self.transform_module:forward(encoder_result)

        -- Create the prediction target from the pre and post sentences
        local prediction = torch.cat(transformed_encoder_result, transformed_encoder_result, 1)
        local target = torch.cat(encoder_pre_result, encoder_post_result, 1)

        local sentence_loss = self.criterion:forward(prediction, target)
        batch_loss = batch_loss + sentence_loss
        data_count = data_count + 1
        -- print('sentence_loss:', sentence_loss)


        -- Starting the backward process
        local sentence_grad = self.criterion:backward(prediction, target)

        local sentence_grad_to_target = torch.mul(sentence_grad, -1)


        local transform_grads_from_pre, transform_grads_from_post
        transform_grads_from_pre = sentence_grad:narrow(1, 1, encoder_result:size(1))
        transform_grads_from_post = sentence_grad:narrow(1, encoder_result:size(1)+1,
          encoder_result:size(1))

        local transform_grads_to_pre, transform_grads_to_post
        transform_grads_to_pre = sentence_grad_to_target:narrow(1, 1, encoder_pre_result:size(1))
        transform_grads_to_post = sentence_grad_to_target:narrow(1, encoder_pre_result:size(1)+1,
          encoder_post_result:size(1))

        local encoder_output_grads = self.transform_module:backward(encoder_result,
          torch.add(transform_grads_from_pre, transform_grads_from_post))

        local encoder_input_grads = self:encoder_backward(embedding_sentence, encoder_output_grads, self.encoder)
        local pre_encoder_input_grads = self:encoder_backward(pre_sentence, transform_grads_to_pre, self.encoder_pre)
        local post_encoder_input_grads = self:encoder_backward(post_sentence, transform_grads_to_post, self.encoder_post)

        if self.update_word_embedding ==1 then
          self:input_module_backward(embedding_sentence_with_vocab_idx,
            pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx,
            encoder_input_grads, pre_encoder_input_grads, post_encoder_input_grads)
        end

        ::continue::
      end -- Finished
      train_loss = train_loss + batch_loss
      print('Batch loss:', batch_loss/batch_size)
      self.progress_writer:write_string(
        string.format('%s: %.4f\n',   'Batch Loss', batch_loss/batch_size))
      self.grad_params:div(batch_size)

      -- Gradient clipping:
      -- if the norm of encoder or decoder gradient is bigger than threshold
      -- scale the gradient to
      local encoder_grad_params = self.grad_params:narrow(1,1,self.encoder_params_element_number)
      local encoder_grad_norm = torch.norm(encoder_grad_params)
      if encoder_grad_norm > self.grad_clip then
        print('clipping gradient for encoder')
          encoder_grad_params:div(encoder_grad_norm/self.grad_clip)
      end

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

function ContextEncoder:load_input_sentences(idx, dataset, corpus)
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

function ContextEncoder:input_module_forward(embedding_sentence_with_vocab_idx,
  pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx)
  local embedding_sentence, pre_sentence, post_sentence
  -- If any of the pre sentence or post sentence contains only the EOS token, skip this datapoint
  if pre_sentence_with_vocab_idx:size(1)<2 or post_sentence_with_vocab_idx:size(1)<2 then
    return
  end

  -- Concatenate three sentences as input for the network,
  -- and map the vocab index in those sentences to the embedding vectors.
  local merged_index = torch.cat(embedding_sentence_with_vocab_idx,
    pre_sentence_with_vocab_idx,1):cat(post_sentence_with_vocab_idx,1)
  local forwardResult = self.input_module:forward(merged_index)

  -- Initialze each sentence with its token mapped to embedding vectors
  embedding_sentence = forwardResult:narrow(1,1,embedding_sentence_with_vocab_idx:size(1))
  pre_sentence = forwardResult:narrow(1,embedding_sentence_with_vocab_idx:size(1) + 1,
    pre_sentence_with_vocab_idx:size(1))
  post_sentence = forwardResult:narrow(1,
    embedding_sentence_with_vocab_idx:size(1) + pre_sentence_with_vocab_idx:size(1) + 1,
    post_sentence_with_vocab_idx:size(1))

  return embedding_sentence, pre_sentence, post_sentence
end

function ContextEncoder:input_module_backward(embedding_sentence_with_vocab_idx,
  pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx,
  encoder_input_grads, pre_encoder_input_grads, post_encoder_input_grads)
  local merged_index = torch.cat(embedding_sentence_with_vocab_idx,
    pre_sentence_with_vocab_idx,1):cat(post_sentence_with_vocab_idx,1)
  local grad_for_input_model = torch.zeros(merged_index:size(1), encoder_input_grads:size(2))

  for i=1, embedding_sentence_with_vocab_idx:size(1) do
    grad_for_input_model[i] = encoder_input_grads[i]
  end
  local count = 1
  for i=embedding_sentence_with_vocab_idx:size(1)+1,  embedding_sentence_with_vocab_idx:size(1)+pre_sentence_with_vocab_idx:size(1) do
    grad_for_input_model[i] = pre_encoder_input_grads[count]
    count = count + 1
  end
  count = 1
  for i=embedding_sentence_with_vocab_idx:size(1)+pre_sentence_with_vocab_idx:size(1)+1,
    embedding_sentence_with_vocab_idx:size(1)+pre_sentence_with_vocab_idx:size(1)+post_encoder_input_grads:size(1)  do
    grad_for_input_model[i] = post_encoder_input_grads[count]
    count = count + 1
  end

  self.input_module:backward(merged_index, grad_for_input_model)
end

function ContextEncoder:encoder_forward(embedding_sentence, encoder)
  local encode_result = encoder:forward(embedding_sentence)
  local output_for_decoder
  if self.encoder.num_layers>1 then
    if string.starts(self.encoder.structure,'bi') then
      -- Case: more than one layer, bi-direction
      local outputs_forward = encode_result[1]
      local outputs_backward = encode_result[2]
      local output_forward_flatten = {}
      local output_backward_flatten = {}
      for l = 1, self.encoder.num_layers do
        if l ==1 then
          output_forward_flatten = outputs_forward[l]
          output_backward_flatten = outputs_backward[l]
        else
          output_forward_flatten = torch.cat(output_forward_flatten,
            outputs_forward[l], 1)
          output_backward_flatten = torch.cat(output_backward_flatten,
            outputs_backward[l], 1)
        end
      end
      output_for_decoder = torch.cat(output_forward_flatten, output_backward_flatten, 1)
    else -- Case: more than one layer, single direction
      for l = 1, self.encoder.num_layers do
        local output_for_current_layer = encode_result[l]
        if l ==1 then
          output_for_decoder = output_for_current_layer
        else
          output_for_decoder = torch.cat(output_for_decoder,output_for_current_layer, 1)
        end
      end
    end
  else
    if string.starts(self.encoder.structure,'bi') then
      -- Case: one layer, bi-direction
      output_for_decoder = torch.cat(encode_result[1], encode_result[2], 1)
    else
      output_for_decoder = encode_result
    end
  end
  return output_for_decoder
end

function ContextEncoder:encoder_backward(embedding_sentence, encoder_output_grads, encoder)
  local grad_input
  if encoder_output_grads ~= nil then
    if self.encoder_num_layers>1 then
      local grad_for_encoder = {}
      local grad_for_encoder_forward = {}
      local grad_for_encoder_backward = {}

      for l = 1, self.encoder_num_layers do
        if string.starts(self.encoder_structure,'bi') then
          grad_for_encoder_forward[l] = encoder_output_grads:narrow(1,
            1 + (l-1)*self.encoder_hidden_dim, self.encoder_hidden_dim)
          grad_for_encoder_backward[l] = encoder_output_grads:narrow(1,
            1 + (l-1)*self.encoder_hidden_dim + self.encoder_num_layers*self.encoder_hidden_dim,
            self.encoder_hidden_dim)
        else
          grad_for_encoder[l] = gradInput_cri:narrow(1, 1 + (l-1)*self.encoder_hidden_dim,
            self.encoder_hidden_dim)
        end
      end
      if string.starts(self.encoder_structure,'bi') then
        grad_for_encoder = {grad_for_encoder_forward, grad_for_encoder_backward}
      end  -- Finished assemble grad_for_encoder
      grad_input = encoder:backward(embedding_sentence, grad_for_encoder)

    else -- Case: self.encoder_num_layers == 1
      local grad_for_encoder = {}
      if string.starts(self.encoder_structure,'bi') then
        grad_for_encoder = {encoder_output_grads:narrow(1, 1, self.encoder_hidden_dim),
          encoder_output_grads:narrow(1, self.encoder_hidden_dim+1, self.encoder_hidden_dim)}
        grad_input = encoder:backward(embedding_sentence, grad_for_encoder)
      else
        grad_input = encoder:backward(embedding_sentence, encoder_output_grads)
      end
    end
  end
  return grad_input
end

function ContextEncoder:calcluate_loss(dataset, corpus)
  self.encoder:evaluate()
  self.encoder_pre:evaluate()
  self.encoder_post:evaluate()

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

function ContextEncoder:calculate_loss_one_instance(idx, dataset, corpus)
  -- load sentence tuple for the current training data point from the corpus
  local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx =
    self:load_input_sentences(idx, dataset, corpus)
  if embedding_sentence_with_vocab_idx == nil or
    pre_sentence_with_vocab_idx == nil or
    post_sentence_with_vocab_idx == nil then
      print('Sentence Loading error for index:')
      print(idx)
      return -1
  end


  -- Initialze each sentence with its token mapped to embedding vectors
  -- Initialze each sentence with its token mapped to embedding vectors
  local embedding_sentence, pre_sentence, post_sentence =
    self:input_module_forward(embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx,
    post_sentence_with_vocab_idx)

  if embedding_sentence == nil or pre_sentence == nil or post_sentence == nil then
    print('Sentence too short.')
    return -1
  end

  -- Start the forward process
  local encoder_result = self:encoder_forward(embedding_sentence, self.encoder)
  local encoder_pre_result = self:encoder_forward(pre_sentence, self.encoder_pre)
  local encoder_post_result = self:encoder_forward(post_sentence, self.encoder_post)

  -- Concatenate encoder_result for pre and post sentence
  local transformed_encoder_result = self.transform_module:forward(encoder_result)

  -- Create the prediction target from the pre and post sentences
  local prediction = torch.cat(transformed_encoder_result, transformed_encoder_result, 1)
  local target = torch.cat(encoder_pre_result, encoder_post_result, 1)

  local sentence_loss = self.criterion:forward(prediction, target)
  -- print('Sentence Loss:', sentence_loss)
  -- Important: to clear the grad_input from the last forward step.
  self.encoder:forget()
  self.encoder_pre:forget()
  self.encoder_post:forget()

  return sentence_loss
end

function ContextEncoder:print_config()
  print('Configurations for the Skip Thoughts Model')
  local num_params = self.params:nElement()
  printf('%-25s = %.2e\n', 'initial learning rate', self.learning_rate)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %f\n',   'gradient clipping', self.grad_clip)
  printf('%-25s = %.2e\n',  'regularization strength', self.reg)
  printf('\n')
  printf('%-25s = %d\n',   'Word Embedding dim', self.emb_dim)
  printf('%-25s = %d\n',   'Word Embedding Update Flag', self.update_word_embedding)
  printf('%-25s = %s\n',   'Encoder structure', self.encoder_structure)
  printf('%-25s = %d\n',   'Encoder hidden dim', self.encoder_hidden_dim)
  printf('%-25s = %d\n',   'Encoder # of layers', self.encoder_num_layers)
  printf('%-25s = %d\n',   'num params', num_params)
end


--
-- Serialization
--
function ContextEncoder:save(path)
  local config = {
    encoder_hidden_dim  = self.encoder_hidden_dim,
    encoder_num_layers  = self.encoder_num_layers,
    encoder_structure   = self.encoder_structure,

    learning_rate       = self.learning_rate,
    batch_size          = self.batch_size,
    reg                 = self.reg,
    grad_clip           = self.grad_clip,

    -- word embedding
    emb_vecs            = self.emb_vecs,
    emb_dim             = self.emb_dim,

    update_word_embedding = self.update_word_embedding

  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function ContextEncoder.load(path)
  local state = torch.load(path)
  local model = sentenceembedding.ContextEncoder.new(state.config)
  model.params:copy(state.params)
  return model
end

return ContextEncoder
