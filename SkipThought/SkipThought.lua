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
  local encoder_config = {
    hidden_dim = self.encoder_hidden_dim,
    num_layers = self.encoder_num_layers,
    emb_vecs   = self.emb_vecs,
    structure  = self.encoder_structure
  }

  self.encoder = SentenceEmbedding.Encoder(encoder_config)

  local decoder_config = {
    encoder_out_dim = self.encoder_hidden_dim,
    in_dim          = self.emb_vecs:size(2),
    hidden_dim      = self.decoder_hidden_dim,
    num_layers      = self.decoder_num_layers
  }

  self.decoder_pre  = SentenceEmbedding.GRUDecoder(decoder_config)
  self.decoder_post = SentenceEmbedding.GRUDecoder(decoder_config)

  -- probability module???????????????????????????????????????????????
  -----------self:new_prob_module()

  -- share_params(self.decoder_pre, self.decoder_post)
end

function SkipThought:new_prob_module()
  local input_dim = self.decoder_hidden_dim
  local input = nn.Identity()()

  local prob_module = nn.Sequential()
    :add(input)
    :add(nn.Linear(input_dim, self.emb_vecs:size(1)))
    :add(nn.LogSoftMax())
  return prob_module
end

function SkipThought:train(dataset, artifact)
  self.encoder:training()
  self.decoder_pre:training()
  self.decoder_post:training()

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.hidden_dim)
  local train_loss = 0
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    -- get target distributions for batch
    local targets = torch.zeros(batch_size)
    for j = 1, batch_size do
      targets[j] = dataset.labels[indices[i + j - 1]]
    end
    local count = 0

    local feval = function(x)
      if x ~= self.params then
        self.params:copy(x)
      end
      self.grad_params:zero()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local linputs, rinputs

        -- load input artifact content using id
        if artifact.src_artfs_ids[dataset.lsents[idx]]~= nil then
          linputs = artifact.src_artfs[artifact.src_artfs_ids[dataset.lsents[idx]]]
        else
          print('Cannot find source:', dataset.lsents[idx])
          break
        end
        if artifact.trg_artfs_ids[dataset.rsents[idx]]~= nil then
          rinputs = artifact.trg_artfs[artifact.trg_artfs_ids[dataset.rsents[idx]]]
        else
          print('Cannot find target:', rsents[idx])
          break
        end
         -- get sentence representations
        local inputs
        if not string.starts(self.structure,'bi') then
          inputs = {self.lrnn:forward(linputs), self.rrnn:forward(rinputs)}
        elseif  string.starts(self.structure,'bi') then
          inputs = {
            self.lrnn:forward(linputs),
            self.lrnn_b:forward(linputs, true), -- true => reverse
            self.rrnn:forward(rinputs),
            self.rrnn_b:forward(rinputs, true)
          }
        end

        -- compute relatedness
        local output = self.sim_module:forward(inputs)

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])

        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        local rep_grad = self.sim_module:backward(inputs, sim_grad)
        -- print("Sim grad", sim_grad)

        if not string.starts(self.structure,'bi') then
          local rnn_grad = self:RNN_backward(linputs, rinputs, rep_grad)
          -- print("RNN grad:", rnn_grad)
        elseif  string.starts(self.structure,'bi') then
          self:BiRNN_backward(linputs, rinputs, rep_grad)
        end
      end
      train_loss = train_loss + loss

      loss = loss / batch_size
      -- print('Loss:', loss)
      self.grad_params:div(batch_size)

      -- Gradient clipping: if the norm of rnn gradient is bigger than threshold
      -- scale the gradient to
      -- local sim_params = self.params:narrow(1,self.rnn_params_element_number, self.params:nElement()-self.rnn_params_element_number)
      -- -- print("sim_params:", sim_params)
      -- local rnn_params = self.params:narrow(1,1,self.rnn_params_element_number)
      -- -- print("rnn_params:", rnn_params)
      -- local sim_grad_params = self.grad_params:narrow(1,self.rnn_params_element_number, self.params:nElement()-self.rnn_params_element_number)
      -- -- print("sim_grad_params:", sim_grad_params)
      local rnn_grad_params = self.grad_params:narrow(1,1,self.rnn_params_element_number)
      -- print("rnn_grad_params:", rnn_grad_params)

      local rnn_grad_norm = torch.norm(rnn_grad_params)
      if rnn_grad_norm > self.grad_clip then
        print('clipping gradient')
          rnn_grad_params:div(rnn_grad_norm/self.grad_clip)
      end

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

-- LSTM backward propagation
function SkipThought:RNN_backward(linputs, rinputs, rep_grad)
  local lgrad, rgrad
  if self.num_layers == 1 then
    lgrad = torch.zeros(linputs:size(1), self.hidden_dim)
    rgrad = torch.zeros(rinputs:size(1), self.hidden_dim)
    lgrad[linputs:size(1)] = rep_grad[1]
    rgrad[rinputs:size(1)] = rep_grad[2]
  else
    lgrad = torch.zeros(linputs:size(1), self.num_layers, self.hidden_dim)
    rgrad = torch.zeros(rinputs:size(1), self.num_layers, self.hidden_dim)
    for l = 1, self.num_layers do
      lgrad[{linputs:size(1), l, {}}] = rep_grad[1][l]
      rgrad[{rinputs:size(1), l, {}}] = rep_grad[2][l]
    end
  end
  self.lrnn:backward(linputs, lgrad)
  local lstm_grad = self.rrnn:backward(rinputs, rgrad)
  return lstm_grad
end

-- Bidirectional LSTM backward propagation
function SkipThought:BiRNN_backward(linputs, rinputs, rep_grad)
  local lgrad, lgrad_b, rgrad, rgrad_b
  if self.num_layers == 1 then
    lgrad   = torch.zeros(linputs:size(1), self.hidden_dim)
    lgrad_b = torch.zeros(linputs:size(1), self.hidden_dim)
    rgrad   = torch.zeros(rinputs:size(1), self.hidden_dim)
    rgrad_b = torch.zeros(rinputs:size(1), self.hidden_dim)
    lgrad[linputs:size(1)] = rep_grad[1]
    rgrad[rinputs:size(1)] = rep_grad[3]
    lgrad_b[1] = rep_grad[2]
    rgrad_b[1] = rep_grad[4]
  else
    lgrad   = torch.zeros(linputs:size(1), self.num_layers, self.hidden_dim)
    lgrad_b = torch.zeros(linputs:size(1), self.num_layers, self.hidden_dim)
    rgrad   = torch.zeros(rinputs:size(1), self.num_layers, self.hidden_dim)
    rgrad_b = torch.zeros(rinputs:size(1), self.num_layers, self.hidden_dim)
    for l = 1, self.num_layers do
      lgrad[{linputs:size(1), l, {}}] = rep_grad[1][l]
      rgrad[{rinputs:size(1), l, {}}] = rep_grad[3][l]
      lgrad_b[{1, l, {}}] = rep_grad[2][l]
      rgrad_b[{1, l, {}}] = rep_grad[4][l]
    end
  end
  self.lrnn:backward(linputs, lgrad)
  self.lrnn_b:backward(linputs, lgrad_b, true)
  self.rrnn:backward(rinputs, rgrad)
  self.rrnn_b:backward(rinputs, rgrad_b, true)
end

-- Predict the similarity of a sentence pair (log probability, should use output:exp() for probability).
function SkipThought:predict(lsent, rsent, artifact)
  self.lrnn:evaluate()
  self.rrnn:evaluate()
  self.sim_module:evaluate()
  local linputs, rinputs
  if artifact.src_artfs_ids[lsent]~= nil then
    linputs = artifact.src_artfs[artifact.src_artfs_ids[lsent]]
  else
    print('Cannot find source:', lsent)
    return nil
  end
  if artifact.trg_artfs_ids[rsent]~= nil then
    rinputs = artifact.trg_artfs[artifact.trg_artfs_ids[rsent]]
  else
    print('Cannot find target:', rsent)
    return nil
  end
  local inputs
  if not  string.starts(self.structure,'bi') then
    inputs = {self.lrnn:forward(linputs), self.rrnn:forward(rinputs)}
  elseif  string.starts(self.structure,'bi') then
    self.lrnn_b:evaluate()
    self.rrnn_b:evaluate()
    inputs = {
      self.lrnn:forward(linputs),
      self.lrnn_b:forward(linputs, true),
      self.rrnn:forward(rinputs),
      self.rrnn_b:forward(rinputs, true)
    }
  end
  local output = self.sim_module:forward(inputs)
  self.lrnn:forget()
  self.rrnn:forget()
  if  string.starts(self.structure,'bi') then
    self.lrnn_b:forget()
    self.rrnn_b:forget()
  end
  -- Jin: bug in original code. range is changed to [1, # of classes]
--  return torch.range(1, self.num_classes):dot(output:exp())
  return output
end

-- Produce similarity predictions for each sentence pair in the dataset.
function SkipThought:predict_dataset(dataset, artifact)
  local predictions = {}
  local targets = dataset.labels
  local loss = 0
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    local output = self:predict(lsent, rsent, artifact)
    predictions[i] = torch.exp(output)
    local example_loss = self.criterion:forward(output, targets[i])
    loss = loss + example_loss
  end
  loss = loss/dataset.size
  return loss, predictions
end

function SkipThought:predict_text(lsent, rsent)
  self.lrnn:evaluate()
  self.rrnn:evaluate()
  self.sim_module:evaluate()
  local linputs, rinputs
  linputs = self.emb_vecs:index(1, lsent:long())
  rinputs = self.emb_vecs:index(1, rsent:long())

  local inputs
  if not  string.starts(self.structure,'bi') then
    inputs = {self.lrnn:forward(linputs), self.rrnn:forward(rinputs)}
  elseif  string.starts(self.structure,'bi') then
    self.lrnn_b:evaluate()
    self.rrnn_b:evaluate()
    inputs = {
      self.lrnn:forward(linputs),
      self.lrnn_b:forward(linputs, true),
      self.rrnn:forward(rinputs),
      self.rrnn_b:forward(rinputs, true)
    }
  end
  local output = self.sim_module:forward(inputs)
  self.lrnn:forget()
  self.rrnn:forget()
  if  string.starts(self.structure,'bi') then
    self.lrnn_b:forget()
    self.rrnn_b:forget()
  end
  return output
end

function SkipThought:print_config()
  local num_params = self.params:nElement()
  local num_sim_params = self:new_sim_module():getParameters():nElement()
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'RNN hidden dim', self.hidden_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %s\n',   'RNN structure', self.structure)
  printf('%-25s = %d\n',   'RNN layers', self.num_layers)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
  printf('%-25s = %d\n',   'Gradient clip', self.grad_clip)
end

--
-- Serialization
--

function SkipThought:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb_vecs,
    learning_rate = self.learning_rate,
    num_layers    = self.num_layers,
    hidden_dim    = self.hidden_dim,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
    grad_clip     = self.grad_clip
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function SkipThought.load(path)
  local state = torch.load(path)
  local model = tracenn.RNNTrace.new(state.config)
  model.params:copy(state.params)
  return model
end
