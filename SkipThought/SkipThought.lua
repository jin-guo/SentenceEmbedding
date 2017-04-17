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
  lookup.weight:copy(config.emb_vecs)
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
    :add(nn.Linear(decoder_config.hidden_dim, self.emb_vecs:size(1)))
    :add(nn.LogSoftMax())
  -- self.prob_module_post = nn.Sequential()
  --   :add(nn.Linear(decoder_config.hidden_dim, self.emb_vecs:size(1)))
  --   :add(nn.LogSoftMax())
  -- share_params(self.prob_module_pre, self.prob_module_post)

  -- For getting all the parameters for the SkipThought model
  local modules = nn.Parallel()
    :add(self.encoder)
    :add(self.decoder_pre)
    :add(self.decoder_post)
    :add(self.prob_module)
  self.params, self.grad_params = modules:getParameters()
  self.encoder_params = self.encoder:parameters()

end

function SkipThought:train(dataset, corpus)
  self.encoder:training()
  self.decoder_pre:training()
  self.decoder_post:training()

  local indices = torch.randperm(dataset.size)
  -- local zeros = torch.zeros(self.hidden_dim)
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
        -- local idx = i + j - 1
        -- if idx ~= 1 then


        -- print('Datapoint ID:' .. idx)

        local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx
        local embedding_sentence, pre_sentence, post_sentence
        local encode_result, pre_decoder_result, post_decoder_result
        local pre_decoder_output, post_decoder_output
        local pre_target, post_target

        -- load sentence tuple for the current training data point from the corpus
        if corpus.ids[dataset.embedding_sentence[idx]]~= nil then
          embedding_sentence_with_vocab_idx = corpus.sentences[corpus.ids[dataset.embedding_sentence[idx]]]
          -- clone the tensor value. Otherwise they will all point to the output of the input_module
          -- local forwardResult = self.input_module:forward(
          --   corpus.sentences[corpus.ids[dataset.embedding_sentence[idx]]])
          -- embedding_sentence = torch.Tensor(forwardResult):copy(forwardResult)
        else
          print('Cannot find embedding sentence for current training '..
            'data point:', dataset.embedding_sentence[idx])
          break
        end

        if corpus.ids[dataset.pre_sentence[idx]]~= nil then
          pre_sentence_with_vocab_idx =
            corpus.sentences[corpus.ids[dataset.pre_sentence[idx]]]
          -- local forwardResult = self.input_module:forward(pre_sentence_with_vocab_idx)
          -- pre_sentence = torch.Tensor(forwardResult):copy(forwardResult)

        else
          print('Cannot find the sentence before the embedding sentence for '..
            'current training data point:', dataset.pre_sentence[idx])
          break
        end

        if corpus.ids[dataset.post_sentence[idx]]~= nil then
          post_sentence_with_vocab_idx =
            corpus.sentences[corpus.ids[dataset.post_sentence[idx]]]
          -- local forwardResult =  self.input_module:forward(post_sentence_with_vocab_idx)
          -- post_sentence = torch.Tensor(forwardResult):copy(forwardResult)
        else
          print('Cannot find the sentence after the embedding sentence for '..
            'current training data point:', dataset.post_sentence[idx])
          break
        end

        if pre_sentence_with_vocab_idx:size(1)<2 or post_sentence_with_vocab_idx:size(1)<2 then
          goto continue
        end

        -- Map the vocab index to the embedding vectors
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

        -- Start the forward process
        encode_result = self.encoder:forward(embedding_sentence)
        pre_decoder_result = self.decoder_pre:forward(pre_sentence,encode_result)
        post_decoder_result = self.decoder_post:forward(post_sentence,encode_result)
        print('pre_decoder_result:size()')
        print(pre_decoder_result:size())

        -- print('decoder_result')
        local decoder_result = torch.cat(pre_decoder_result, post_decoder_result, 1)
        -- print(decoder_result:size())


        if self.decoder_num_layers == 1 then
          decoder_output = self.prob_module:forward(decoder_result)

          -- pre_decoder_output = self.prob_module_pre:forward(pre_decoder_result)
          -- post_decoder_output = self.prob_module_post:forward(post_decoder_result)
        else
          -- If there are more than one layers, using the final layer output as the decoding result
          decoder_output = self.prob_module:forward(
            decoder_result:select(2, decoder_result:size(2)))

          -- pre_decoder_output = self.prob_module_pre:forward(
          --   pre_decoder_result:select(2, pre_decoder_result:size(2)))
          -- post_decoder_output = self.prob_module_post:forward(
          --   post_decoder_result:select(2, post_decoder_result:size(2)))
        end

        pre_target = pre_sentence_with_vocab_idx:sub(2, -1)
        post_target = post_sentence_with_vocab_idx:sub(2, -1)
        local target = torch.cat(pre_target, post_target, 1)
        local sentence_loss = self.criterion:forward(decoder_output, target)
        local sentence_grad = self.criterion:backward(decoder_output, target)

        train_loss = train_loss + sentence_loss
        print(sentence_loss)

        local encoder_output_grads = nil
        if self.decoder_num_layers == 1 then
          -- print('Before here')
          prob_grad = self.prob_module:backward(decoder_result, sentence_grad)
          -- print('After here')
        else
          prob_grad = self.prob_module:backward(decoder_result:select(2, decoder_result:size(2)), sentence_grad)
        end

        local pre_prob_grad, post_prob_grad
        pre_prob_grad = prob_grad:narrow(1, 1, pre_decoder_result:size(1))
        post_prob_grad = prob_grad:narrow(1, pre_decoder_result:size(1)+1,
          post_decoder_result:size(1))

        local pre_decoder_input_grad, pre_encoder_output_grads = self.decoder_pre:backward(pre_sentence, pre_prob_grad)
        encoder_output_grads = torch.Tensor(pre_encoder_output_grads):copy(pre_encoder_output_grads)
        local post_decoder_input_grad, post_encoder_output_grads = self.decoder_post:backward(post_sentence, post_prob_grad)
        encoder_output_grads:add(post_encoder_output_grads)

        if encoder_output_grads ~= nil then
          local encode_grad = self.encoder:backward(embedding_sentence, encoder_output_grads)
        end
        ::continue::
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
