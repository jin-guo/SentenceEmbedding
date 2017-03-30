local SkipThought = torch.class('SentenceEmbedding.SkipThought')

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
  loopup.weight:copy(config.emb_vecs)
  self.input_module:add(lookup)


  -- initialize Encoder model
  local encoder_config = {
    hidden_dim = self.encoder_hidden_dim,
    num_layers = self.encoder_num_layers,
    emb_vecs   = self.emb_vecs,
    structure  = self.encoder_structure
  }
  self.encoder = SentenceEmbedding.Encoder(encoder_config)

  -- initialize Decoder model
  local decoder_config = {
    encoder_out_dim = self.encoder_hidden_dim,
    in_dim          = self.emb_vecs:size(2),
    hidden_dim      = self.decoder_hidden_dim,
    num_layers      = self.decoder_num_layers
  }

  self.decoder_pre  = SentenceEmbedding.GRUDecoder(decoder_config)
  self.decoder_post = SentenceEmbedding.GRUDecoder(decoder_config)

  -- initialize Probability model
  self.prob_module = nn.Sequential()
    :add(nn.Linear(decoder_config.hidden_dim, vecs:size(1)))
    :add(nn.LogSoftMax())

  -- For getting all the parameters for the SkipThought model
  local modules = nn.Parallel()
    :add(encoder)
    :add(decoder_pre)
    :add(decoder_post)
    :add(prob_module)
  self.params, self.grad_params = modules:getParameters()
  self.encoder_params = encoder:parameters()

end

function SkipThought:train(dataset, corpus)
  self.encoder:training()
  self.decoder_pre:training()
  self.decoder_post:training()

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.hidden_dim)
  local train_loss = 0
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      if x ~= self.params then
        self.params:copy(x)
      end
      self.grad_params:zero()

      local loss = 0
      -- For each datapoint in current batch
      for j = 1, batch_size do
        local idx = indices[i + j - 1]

        local embedding_sentence, pre_sentence, post_sentence
        local encode_result, pre_decoder_result, post_decoder_result
        local pre_decoder_output, post_decoder_output
        local pre_target, post_target

        -- load sentence tuple for the current training data point from the corpus
        if corpus.sentences[dataset.embedding_sentence[idx]]~= nil then
          embedding_sentence = self.input_module:forward(
            corpus.sentences[dataset.embedding_sentence[idx]])
        else
          print('Cannot find embedding sentence for current training data point:', dataset.embedding_sentence[idx])
          break
        end

        if corpus.sentences[dataset.pre_sentence[idx]]~= nil then
          pre_sentence = self.input_module:forward(
            corpus.sentences[dataset.pre_sentence[idx]])
        else
          print('Cannot find the sentence before the embedding sentence for current training data point:', dataset.pre_sentence[idx])
          break
        end

        if corpus.sentences[dataset.post_sentence[idx]]~= nil then
          post_sentence = self.input_module:forward(
            corpus.sentences[dataset.post_sentence[idx]])
        else
          print('Cannot find the sentence after the embedding sentence for current training data point:', dataset.post_sentence[idx])
          break
        end

        -- Start the forward process
        encode_result = encoder:forward(embedding_sentence)
        pre_decoder_result = decoder_pre:forward(pre_sentence,encode_result)
        post_decoder_result = decoder_post:forward(post_sentence,encode_result)


        if self.decoder_num_layers.num_layers == 1 then
          pre_decoder_output = prob_module:forward(pre_decoder_result)
          post_decoder_output = prob_module:forward(post_decoder_result)
        else
          -- If there are more than one layers, using the final layer output as the decoding result
          pre_decoder_output = prob_module:forward(pre_decoder_result:select(3, pre_decoder_result:size(3)))
          post_decoder_output = prob_module:forward(post_decoder_result:select(3, pre_decoder_result:size(3)))
        end

        -- Remove the last output token since the EOS should be predicted before this token
        pre_decoder_output = pre_decoder_output:sub(1, pre_decoder_output:size(1)-1)
        post_decoder_output = post_decoder_output:sub(1, post_decoder_output:size(1)-1)


        -- The prediction target starts from the second of the token in the sentence sequence
        pre_target = corpus.sentences[dataset.pre_sentence[idx]]:sub(2, corpus.sentences[dataset.pre_sentence[idx]]:size(1))
        post_target = corpus.sentences[dataset.post_sentence[idx]]:sub(2, corpus.sentences[dataset.post_sentence[idx]]:size(1))



        -- compute loss and backpropagate
        local pre_sentence_loss = self.criterion:forward(pre_decoder_output, pre_target)
        local pre_sentence_grad = self.criterion:backward(pre_decoder_output, pre_target)

        local post_sentence_loss = self.criterion:forward(post_decoder_output, post_target)
        local post_sentence_grad = self.criterion:backward(post_decoder_output, post_target)

        loss = pre_sentence_loss + post_sentence_loss


        local pre_prob_grad, post_prob_grad
        if decoder_config.num_layers == 1 then
          pre_prob_grad = self.prob_module:backward(pre_decoder_output, pre_sentence_grad)
          post_prob_grad = self.prob_module:backward(post_decoder_output, post_sentence_grad)
        else
          pre_prob_grad = self.prob_module:backward(pre_decoder_output:select(3, pre_decoder_result:size(3)), pre_sentence_grad)
          pre_prob_grad = self.prob_module:backward(pre_decoder_output:select(3, pre_decoder_result:size(3)), post_sentence_grad)
        end

        local pre_decoder_input_grad, pre_encoder_output_grads = decoder_pre:backward(pre_sentence, pre_prob_grad)
        local post_decoder_input_grad, post_encoder_output_grads = decoder_pre:backward(post_sentence, post_prob_grad)

        local encoder_output_grads
        encoder_output_grads:add(pre_encoder_output_grads, post_encoder_output_grads)
        local encode_grad = encoder:backward(embedding_sentence, encoder_output_grads)


      end
      train_loss = train_loss + loss

      loss = loss / batch_size
      -- print('Loss:', loss)
      self.grad_params:div(batch_size)

      -- TODO: Gradient clipping
      -- : if the norm of rnn gradient is bigger than threshold
      -- scale the gradient to
      -- local encoder_grad_params = self.grad_params:narrow(1,1,self.rnn_params_element_number)
      --
      -- local rnn_grad_norm = torch.norm(rnn_grad_params)
      -- if rnn_grad_norm > self.grad_clip then
      --   print('clipping gradient')
      --     rnn_grad_params:div(rnn_grad_norm/self.grad_clip)
      -- end

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2 * batch_size/dataset.size
      -- Final derivatives to return after regularization:
      -- self.grad_params + self.reg*self.params
      self.grad_params:add(self.reg, self.params)
      -- count = count + 1
      -- print(count)
      return loss, self.grad_params
    end

  --  print('Check the gradients:', self.grad_params:size(1)*2)
  --  diff, dc, dc_est = optim.checkgrad(feval, self.params:clone())
  --  print('Diff must be close to 1e-8: diff = ' .. diff)

    optim.rmsprop(feval, self.params, self.optim_state)
  end

  train_loss = train_loss/dataset.size
  xlua.progress(dataset.size, dataset.size)
  print('Training loss', train_loss)
  return train_loss
end
