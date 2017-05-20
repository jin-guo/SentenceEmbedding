local AutoEncoder = torch.class('sentenceembedding.AutoEncoder')

function AutoEncoder:__init(config)
  self.name                  = 'AutoEncoder'
  self.encoder_hidden_dim    = config.encoder_hidden_dim    or 50
  self.encoder_num_layers    = config.encoder_num_layers    or 1
  self.encoder_structure     = config.encoder_structure     or 'bigru'
  self.decoder_hidden_dim    = config.decoder_hidden_dim    or 50
  self.decoder_num_layers    = config.decoder_num_layers    or 1

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

  self.decoder  = sentenceembedding.GRUDecoder(decoder_config)

  -- initialize Probability model
  self.prob_module = nn.Sequential()
    :add(nn.Linear(self.decoder_num_layers*self.decoder_hidden_dim, self.emb_vecs:size(1)))
    :add(nn.LogSoftMax())

  -- For getting all the parameters for the SkipThought model
  local modules = nn.Parallel()
    :add(self.encoder)
    :add(self.decoder)
    :add(self.prob_module)

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

  -- Get the number of parameters for decoder
  -- (same configuration for pre and post decoder, so same number)
  self.decoder_params = self.decoder:parameters()
  self.decoder_params_element_number = 0
  for i=1,#self.decoder_params do
    self.decoder_params_element_number =
      self.decoder_params_element_number + self.decoder_params[i]:nElement()
  end
end

function AutoEncoder:train(dataset, corpus)
  self.encoder:training()
  self.decoder:training()
  self.prob_module:training()

  local indices = torch.randperm(dataset.size)
  local train_loss = 0
  local data_count = 0
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
        local embedding_sentence_with_vocab_idx =
          self:load_input_sentences(idx, dataset, corpus)
        if embedding_sentence_with_vocab_idx == nil then
          print('Sentence Loading error for index:')
          print(idx)
          goto continue
        elseif embedding_sentence_with_vocab_idx:size(1)<2 then
          print('Sentence too short.')
          goto continue
        end

        -- Initialze sentence with its token mapped to embedding vectors
        local embedding_sentence = self.input_module:forward(embedding_sentence_with_vocab_idx)
        -- Remove the last token (EOS) as the input for the decoder
        local embedding_sentence_for_decoder_input = embedding_sentence:sub(1,-2)

        -- Start the forward process
        local output_for_decoder = self:encoder_forward(embedding_sentence)

        -- Forward result to Decoder
        local decoder_result = self.decoder:forward(embedding_sentence_for_decoder_input, output_for_decoder)

        local decoder_output = self.prob_module:forward(decoder_result)

        -- Create the prediction target from the embedding sentences
        local target = embedding_sentence_with_vocab_idx:sub(2, -1)

        local sentence_loss = self.criterion:forward(decoder_output, target)
        batch_loss = batch_loss + sentence_loss
        data_count = data_count + 1
        -- print('sentence_loss:', sentence_loss)

        -- Starting the backward process
        local sentence_grad = self.criterion:backward(decoder_output, target)
        local prob_grad = self.prob_module:backward(decoder_result, sentence_grad)

        -- Get the gradient for sentence decoder
        local encoder_output_grads =
          self:decoder_backward(embedding_sentence_for_decoder_input, prob_grad)


        local encoder_input_grads = self:encoder_backward(embedding_sentence, encoder_output_grads)

        if self.update_word_embedding ==1 then
            self.input_module:backward(embedding_sentence_with_vocab_idx, encoder_input_grads)
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

      local decoder_grad_params = self.grad_params:narrow(1,
        self.encoder_params_element_number+1, self.decoder_params_element_number)
      local decoder_grad_norm = torch.norm(decoder_grad_params)
      if decoder_grad_norm > self.grad_clip then
        print('clipping gradient for pre decoder')
          decoder_grad_params:div(decoder_grad_norm/self.grad_clip)
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

function AutoEncoder:load_input_sentences(idx, dataset, corpus)
  local embedding_sentence_with_vocab_idx
  -- load sentence tuple for the current training data point from the corpus
  if corpus.ids[dataset.embedding_sentence[idx]]~= nil then
    embedding_sentence_with_vocab_idx = corpus.sentences[corpus.ids[dataset.embedding_sentence[idx]]]
  else
    print('Cannot find embedding sentence for current training '..
      'data point:', dataset.embedding_sentence[idx])
    return
  end
  return embedding_sentence_with_vocab_idx
end

function AutoEncoder:encoder_forward(embedding_sentence)
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

function AutoEncoder:encoder_backward(embedding_sentence, encoder_output_grads)
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

function AutoEncoder:decoder_backward(embedding_sentence_for_decoder_input, prob_grad)
  local decoder_input_grad, encoder_output_grads
  if self.decoder_hidden_dim>1 then
    local gradInput_for_decoder = torch.zeros(embedding_sentence_for_decoder_input:size(1),
      self.decoder_num_layers, self.decoder_hidden_dim)
    for t = 1, embedding_sentence_for_decoder_input:size(1) do
      local gradInput_current_time = prob_grad:select(1,t)
      gradInput_current_time = gradInput_current_time:resize(self.decoder_num_layers, self.decoder_hidden_dim)
      gradInput_for_decoder[t] = gradInput_current_time
    end
    decoder_input_grad, encoder_output_grads = self.decoder:backward(embedding_sentence_for_decoder_input, gradInput_for_decoder)

  else -- Case: self.decoder_num_layers == 1
    decoder_input_grad, encoder_output_grads = self.decoder:backward(embedding_sentence_for_decoder_input, prob_grad)
  end
  return encoder_output_grads
end

function AutoEncoder:calcluate_loss(dataset, corpus)
  self.encoder:evaluate()
  self.decoder:evaluate()
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

function AutoEncoder:calculate_loss_one_instance(idx, dataset, corpus)
  -- load sentence tuple for the current training data point from the corpus
  local embedding_sentence_with_vocab_idx =
    self:load_input_sentences(idx, dataset, corpus)
  if embedding_sentence_with_vocab_idx == nil then
    print('Sentence Loading error for index:')
    print(idx)
    return -1
  elseif embedding_sentence_with_vocab_idx:size(1)<2 then
    print('Sentence too short.')
    return -1
  end

  -- Initialze sentence with its token mapped to embedding vectors
  local embedding_sentence = self.input_module:forward(embedding_sentence_with_vocab_idx)
  -- Remove the last token (EOS) as the input for the decoder
  local embedding_sentence_for_decoder_input = embedding_sentence:sub(1,-2)

  -- Start the forward process
  local output_for_decoder = self:encoder_forward(embedding_sentence)

  -- Forward result to Decoder
  local decoder_result = self.decoder:forward(embedding_sentence_for_decoder_input,output_for_decoder)

  local decoder_output = self.prob_module:forward(decoder_result)

  -- Create the prediction target from the embedding sentences
  local target = embedding_sentence_with_vocab_idx:sub(2, -1)

  local sentence_loss = self.criterion:forward(decoder_output, target)
  -- print('Sentence Loss:', sentence_loss)
  -- Important: to clear the grad_input from the last forward step.
  self.encoder:forget()
  self.decoder:forget()

  return sentence_loss
end

function AutoEncoder:print_config()
  print('Configurations for the AutoEncoder Model')
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
  printf('%-25s = %d\n',   'Decoder hidden dim', self.decoder_hidden_dim)
  printf('%-25s = %d\n',   'Decoder # of layers', self.decoder_num_layers)
  printf('%-25s = %d\n',   'num params', num_params)
end


--
-- Serialization
--
function AutoEncoder:save(path)
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
    emb_dim             = self.emb_dim,

    update_word_embedding = self.update_word_embedding

  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function AutoEncoder.load(path)
  local state = torch.load(path)
  local model = sentenceembedding.AutoEncoder.new(state.config)
  model.params:copy(state.params)
  return model
end

return AutoEncoder
