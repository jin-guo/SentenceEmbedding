local SkipThought = torch.class('sentenceembedding.SkipThought')

function SkipThought:__init(config)
  self.encoder_hidden_dim    = config.encoder_hidden_dim    or 50
  self.encoder_num_layers    = config.encoder_num_layers    or 1
  self.encoder_structure     = config.encoder_structure     or 'bigru'
  self.decoder_hidden_dim    = config.decoder_hidden_dim    or 50
  self.decoder_num_layers    = config.decoder_num_layers    or 1

  self.learning_rate = config.learning_rate or 0.0001
  self.batch_size    = config.batch_size    or 10
  self.reg           = config.reg           or 0
  self.grad_clip     = config.grad_clip     or 10

  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

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


  -- initialize Encoder model
  local encoder_config = {
    hidden_dim = self.encoder_hidden_dim,
    num_layers = self.encoder_num_layers,
    emb_vecs   = self.emb_vecs,
    structure  = self.encoder_structure
  }
  self.encoder = sentenceembedding.Encoder(encoder_config)

  -- initialize Decoder model
  local encoder_out_dim_real
  if string.starts(self.encoder_structure,'bi') then
    encoder_out_dim_real = 2*self.encoder_num_layers*self.encoder_hidden_dim
  else
    encoder_out_dim_real = self.encoder_num_layers*self.encoder_hidden_dim
  end
  local decoder_config = {
    encoder_out_dim = encoder_out_dim_real,
    in_dim          = self.emb_vecs:size(2),
    hidden_dim      = self.decoder_hidden_dim,
    num_layers      = self.decoder_num_layers
  }

  self.decoder_pre  = sentenceembedding.GRUDecoder(decoder_config)
  self.decoder_post = sentenceembedding.GRUDecoder(decoder_config)

  -- initialize Probability model
  self.prob_module = nn.Sequential()
    :add(nn.Linear(self.decoder_num_layers*self.decoder_hidden_dim, self.emb_vecs:size(1)))
    :add(nn.LogSoftMax())

  -- For getting all the parameters for the SkipThought model
  local modules = nn.Parallel()
    :add(self.encoder)
    :add(self.decoder_pre)
    :add(self.decoder_post)
    :add(self.prob_module)
    -- :add(self.input_module)
  self.params, self.grad_params = modules:getParameters()

  -- Get the number of parameters for encoder
  self.encoder_params = self.encoder:parameters()
  self.encoder_params_element_number = 0
  for i=1,#self.encoder_params do
    self.encoder_params_element_number =
      self.encoder_params_element_number + self.encoder_params[i]:nElement()
  end

  -- Get the number of parameters for decoder
  -- (same configuration for pre and post decoder, so same number)
  self.decoder_params = self.encoder:parameters()
  self.decoder_params_element_number = 0
  for i=1,#self.decoder_params do
    self.decoder_params_element_number =
      self.decoder_params_element_number + self.decoder_params[i]:nElement()
  end
end

function SkipThought:train(dataset, corpus)
  self.encoder:training()
  self.decoder_pre:training()
  self.decoder_post:training()
  self.prob_module:training()

  local indices = torch.randperm(dataset.size)
  local train_loss = 0
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

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
        local embedding_sentence, pre_sentence, post_sentence =
          self:input_module_forward(embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx,
          post_sentence_with_vocab_idx)

        if embedding_sentence == nil or pre_sentence == nil or post_sentence == nil then
          print('Sentence too short.')
          goto continue
        end

        -- Start the forward process
        local output_for_decoder = self:encoder_forward(embedding_sentence)

        -- Forward result to Decoder
        local decoder_result = self:decoder_forward(pre_sentence, post_sentence, output_for_decoder)

        local decoder_output = self.prob_module:forward(decoder_result)

        -- Create the prediction target from the pre and post sentences
        local pre_target, post_target
        pre_target = pre_sentence_with_vocab_idx:sub(2, -1)
        post_target = post_sentence_with_vocab_idx:sub(2, -1)
        local target = torch.cat(pre_target, post_target, 1)

        local sentence_loss = self.criterion:forward(decoder_output, target)
        batch_loss = batch_loss + sentence_loss
        print('sentence_loss:', sentence_loss)

        -- Starting the backward process
        local sentence_grad = self.criterion:backward(decoder_output, target)
        local prob_grad = self.prob_module:backward(decoder_result, sentence_grad)

        -- Get the gradient for the pre sentence decoder and the post sentence decoder
        local pre_encoder_output_grads, post_encoder_output_grads =
          self:decoder_backward(pre_sentence, post_sentence, prob_grad)

        local encoder_output_grads
        encoder_output_grads = torch.Tensor(pre_encoder_output_grads):copy(pre_encoder_output_grads)
        encoder_output_grads:add(post_encoder_output_grads)

        self:encoder_backward(embedding_sentence, encoder_output_grads)

        ::continue::
      end -- Finished
      train_loss = train_loss + batch_loss

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

      local pre_decoder_grad_params = self.grad_params:narrow(1,
        self.encoder_params_element_number+1, self.decoder_params_element_number)
      local pre_decoder_grad_norm = torch.norm(pre_decoder_grad_params)
      if pre_decoder_grad_norm > self.grad_clip then
        print('clipping gradient for pre decoder')
          pre_decoder_grad_params:div(pre_decoder_grad_norm/self.grad_clip)
      end

      local post_decoder_grad_params = self.grad_params:narrow(1,
        self.encoder_params_element_number + self.decoder_params_element_number + 1,
        self.decoder_params_element_number)
      local post_decoder_grad_norm = torch.norm(post_decoder_grad_params)
      if post_decoder_grad_norm > self.grad_clip then
        print('clipping gradient for post decoder')
          post_decoder_grad_params:div(post_decoder_grad_norm/self.grad_clip)
      end

      -- regularization
      batch_loss = batch_loss + 0.5 * self.reg * self.params:norm() ^ 2 * batch_size/dataset.size
      -- Final derivatives to return after regularization:
      -- self.grad_params + self.reg*self.params
      self.grad_params:add(self.reg, self.params)

      return batch_loss, self.grad_params
    end
    -- Works better than optim.adam
    optim.rmsprop(feval, self.params, self.optim_state)
  end

  train_loss = train_loss/dataset.size
  xlua.progress(dataset.size, dataset.size)
  -- print('Training loss', train_loss)
  return train_loss
end

function SkipThought:load_input_sentences(idx, dataset, corpus)
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

function SkipThought:input_module_forward(embedding_sentence_with_vocab_idx,
  pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx)
  local embedding_sentence, pre_sentence, post_sentence
  -- If any of the pre sentence or post sentence contains only the EOS token, skip this datapoint
  if pre_sentence_with_vocab_idx:size(1)<2 or post_sentence_with_vocab_idx:size(1)<2 then
    return
  end

  -- Concatenate three sentences as input for the network,
  -- and map the vocab index in those sentences to the embedding vectors.
  -- For the decoder input, the last token (EOS) of pre and post sentence is removed from the input.
  local merged_index = torch.cat(embedding_sentence_with_vocab_idx,
    pre_sentence_with_vocab_idx:sub(1,-2),1):cat(post_sentence_with_vocab_idx:sub(1,-2),1)
  local forwardResult = self.input_module:forward(merged_index)

  -- Initialze each sentence with its token mapped to embedding vectors
  embedding_sentence = forwardResult:narrow(1,1,embedding_sentence_with_vocab_idx:size(1))
  pre_sentence = forwardResult:narrow(1,embedding_sentence_with_vocab_idx:size(1) + 1,
    pre_sentence_with_vocab_idx:size(1)-1)
  post_sentence = forwardResult:narrow(1,
    embedding_sentence_with_vocab_idx:size(1) + pre_sentence_with_vocab_idx:size(1),
    post_sentence_with_vocab_idx:size(1)-1)

  return embedding_sentence, pre_sentence, post_sentence
end

function SkipThought:encoder_forward(embedding_sentence)
  local encode_result = self.encoder:forward(embedding_sentence)
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

function SkipThought:decoder_forward(pre_sentence, post_sentence, output_for_decoder)
  local pre_decoder_result = self.decoder_pre:forward(pre_sentence,output_for_decoder)
  local post_decoder_result = self.decoder_post:forward(post_sentence,output_for_decoder)

  local decoder_result
  local pre_decoder_result_size, post_decoder_result_size
  if self.decoder_num_layers>1 then
    local pre_decoder_output_flatten = torch.zeros(pre_sentence:size(1),
      self.decoder_num_layers*self.decoder_hidden_dim)
    for t = 1, pre_sentence:size(1) do
      local output_current_time = pre_decoder_result:select(1, t)
      output_current_time = output_current_time:view(output_current_time:nElement())
      pre_decoder_output_flatten[t] = output_current_time
    end

    local post_decoder_output_flatten = torch.zeros(post_sentence:size(1),
      self.decoder_num_layers*self.decoder_hidden_dim)
    for t = 1, post_sentence:size(1) do
      local output_current_time = post_decoder_result:select(1, t)
      output_current_time = output_current_time:view(output_current_time:nElement())
      post_decoder_output_flatten[t] = output_current_time
    end

    -- Concatenate the decoder output from pre and post sentences
    -- as the input for the probability module
    decoder_result = torch.cat(pre_decoder_output_flatten, post_decoder_output_flatten, 1)
  else -- Case: self.decoder_num_layers == 1
    decoder_result = torch.cat(pre_decoder_result, post_decoder_result, 1)
  end
  return decoder_result
end

function SkipThought:encoder_backward(embedding_sentence, encoder_output_grads)
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
      grad_input = self.encoder:backward(embedding_sentence, grad_for_encoder)

    else -- Case: self.encoder_num_layers == 1
      local grad_for_encoder = {}
      if string.starts(self.encoder_structure,'bi') then
        grad_for_encoder = {encoder_output_grads:narrow(1, 1, self.encoder_hidden_dim),
          encoder_output_grads:narrow(1, self.encoder_hidden_dim+1, self.encoder_hidden_dim)}
        grad_input = self.encoder:backward(embedding_sentence, grad_for_encoder)
      else
        grad_input = self.encoder:backward(embedding_sentence, encoder_output_grads)
      end
    end
  end
  return grad_input
end

function SkipThought:decoder_backward(pre_sentence, post_sentence, prob_grad)
  local pre_prob_grad, post_prob_grad
  pre_prob_grad = prob_grad:narrow(1, 1, pre_sentence:size(1))
  post_prob_grad = prob_grad:narrow(1, pre_sentence:size(1)+1,
    post_sentence:size(1))

  local pre_decoder_input_grad, pre_encoder_output_grads
  local post_decoder_input_grad, post_encoder_output_grads
  if self.decoder_hidden_dim>1 then
    local gradInput_for_pre_decoder = torch.zeros(pre_sentence:size(1),
      self.decoder_num_layers, self.decoder_hidden_dim)
    for t = 1, pre_sentence:size(1) do
      local gradInput_current_time = pre_prob_grad:select(1,t)
      gradInput_current_time = gradInput_current_time:resize(self.decoder_num_layers, self.decoder_hidden_dim)
      gradInput_for_pre_decoder[t] = gradInput_current_time
    end
    pre_decoder_input_grad, pre_encoder_output_grads = self.decoder_pre:backward(pre_sentence, gradInput_for_pre_decoder)

    local gradInput_for_post_decoder = torch.zeros(post_sentence:size(1),
      self.decoder_num_layers, self.decoder_hidden_dim)
    for t = 1, post_sentence:size(1) do
      local gradInput_current_time = post_prob_grad:select(1,t)
      gradInput_current_time = gradInput_current_time:resize(self.decoder_num_layers, self.decoder_hidden_dim)
      gradInput_for_post_decoder[t] = gradInput_current_time
    end
    post_decoder_input_grad, post_encoder_output_grads = self.decoder_post:backward(post_sentence, gradInput_for_post_decoder)

  else -- Case: self.decoder_num_layers == 1
    pre_decoder_input_grad, pre_encoder_output_grads = self.decoder_pre:backward(pre_sentence, pre_prob_grad)
    post_decoder_input_grad, post_encoder_output_grads = self.decoder_post:backward(post_sentence, post_prob_grad)
  end
  return pre_encoder_output_grads, post_encoder_output_grads
end

function SkipThought:calcluate_loss(dataset, corpus)
  self.encoder:evaluate()
  self.decoder_pre:evaluate()
  self.decoder_post:evaluate()
  self.prob_module:evaluate()
  local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx
  local embedding_sentence, pre_sentence, post_sentence
  local output_for_decoder
  local decoder_result
  local decoder_output
  local pre_target, post_target, target

  local total_loss = 0
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    -- load sentence tuple for the current training data point from the corpus
    embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx =
      self:load_input_sentences(i, dataset, corpus)
    if embedding_sentence_with_vocab_idx == nil or
      pre_sentence_with_vocab_idx == nil or
      post_sentence_with_vocab_idx == nil then
        print('Sentence Loading error for index:')
        print(idx)
        break
    end


    -- Initialze each sentence with its token mapped to embedding vectors
    embedding_sentence, pre_sentence, post_sentence =
      self:input_module_forward(embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx,
      post_sentence_with_vocab_idx)

    if embedding_sentence == nil or pre_sentence == nil or post_sentence == nil then
      print('Sentence too short.')
      break
    end

    -- Start the forward process
    output_for_decoder = self:encoder_forward(embedding_sentence)

    -- Forward result to Decoder
    decoder_result = self:decoder_forward(pre_sentence, post_sentence, output_for_decoder)

    decoder_output = self.prob_module:forward(decoder_result)

    -- Create the prediction target from the pre and post sentences
    pre_target = pre_sentence_with_vocab_idx:sub(2, -1)
    post_target = post_sentence_with_vocab_idx:sub(2, -1)
    target = torch.cat(pre_target, post_target, 1)

    local sentence_loss = self.criterion:forward(decoder_output, target)
    total_loss = total_loss + sentence_loss
  end
  xlua.progress(dataset.size, dataset.size)
  return total_loss/dataset.size
end

function SkipThought:print_config()
  print('Configurations for the Skip Thoughts Model')
  local num_params = self.params:nElement()
  -- local num_sim_params = self:new_sim_module():getParameters():nElement()
  printf('%-25s = %d\n',   'num params', num_params)
  -- printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  -- printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  -- printf('%-25s = %d\n',   'RNN hidden dim', self.hidden_dim)
  -- printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  -- printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  -- printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  -- printf('%-25s = %s\n',   'RNN structure', self.structure)
  -- printf('%-25s = %d\n',   'RNN layers', self.num_layers)
  -- printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
  -- printf('%-25s = %d\n',   'Gradient clip', self.grad_clip)
end


--
-- Serialization
--
function SkipThought:save(path)
  local config = {
    encoder_hidden_dim  = self.encoder_hidden_dim,
    encoder_num_layers  = self.encoder_num_layers,
    encoder_structure   = self.encoder_structure,
    decoder_hidden_dim  = self.decoder_hidden_dim,
    decoder_num_layers  = self.decoder_num_layers,

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

function SkipThought.load(path)
  local state = torch.load(path)
  local model = sentenceembedding.SkipThought.new(state.config)
  model.params:copy(state.params)
  return model
end

return SkipThought
