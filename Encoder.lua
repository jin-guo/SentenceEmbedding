local Encoder, parent = torch.class('sentenceembedding.Encoder', 'nn.Module')

function Encoder:__init(config)
  parent.__init(self)

  self.hidden_dim    = config.hidden_dim    or 50
  self.num_layers    = config.num_layers    or 1
  self.structure     = config.structure     or 'bigru'

  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- initialize RNN model
  local rnn_config = {
    in_dim = self.emb_dim,
    hidden_dim = self.hidden_dim,
    num_layers = self.num_layers,
    gate_output = true, -- ignored by RNN models other than LSTM
  }

  if self.structure == 'lstm' then
    self.rnn = sentenceembedding.LSTM(rnn_config)
  elseif self.structure == 'bilstm' then
    self.rnn = sentenceembedding.LSTM(rnn_config)
    self.rnn_b = sentenceembedding.LSTM(rnn_config) -- backward RNN
  elseif self.structure == 'irnn' then
    self.rnn = sentenceembedding.IRNN(rnn_config)
  elseif self.structure == 'birnn' then
    self.rnn = sentenceembedding.IRNN(rnn_config)
    self.rnn_b = sentenceembedding.IRNN(rnn_config) -- backward RNN
  elseif self.structure == 'gru' then
    self.rnn = sentenceembedding.GRU(rnn_config)
  elseif self.structure == 'bigru' then
    self.rnn = sentenceembedding.GRU(rnn_config)
    self.rnn_b = sentenceembedding.GRU(rnn_config) -- backward RNN
  else
    error('invalid RNN type: ' .. self.structure)
  end

  -- location of the parameters
  if string.starts(self.structure,'bi') then
    -- tying the forward and backward weights improves performance
    share_params(self.rnn_b, self.rnn)
  end
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- Returns the final hidden state of the GRU.
function Encoder:forward(inputs)
  if not string.starts(self.structure,'bi') then
    return self.rnn:forward(inputs)
  else
    return {self.rnn:forward(inputs), self.rnn_b:forward(inputs, true)}
  end
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x hidden_dim tensor.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function Encoder:backward(inputs, grad_outputs)
  if not string.starts(self.structure,'bi') then
    return self:RNN_backward(inputs, grad_outputs)
  elseif  string.starts(self.structure,'bi') then
    return self:BiRNN_backward(inputs, grad_outputs)
  end
end

-- RNN backward propagation
function Encoder:RNN_backward(inputs, grad_outputs)
  local grad
  if self.num_layers == 1 then
    grad = torch.zeros(inputs:size(1), self.hidden_dim)
    grad[inputs:size(1)] = grad_outputs
  else
    grad = torch.zeros(inputs:size(1), self.num_layers, self.hidden_dim)
    for l = 1, self.num_layers do
      grad[{inputs:size(1), l, {}}] = grad_outputs[l]
    end
  end
  return self.rnn:backward(inputs, grad)
end

-- Bidirectional RNN backward propagation
function Encoder:BiRNN_backward(inputs, grad_outputs)
  local grad, grad_b
  if self.num_layers == 1 then
    grad   = torch.zeros(inputs:size(1), self.hidden_dim)
    grad_b = torch.zeros(inputs:size(1), self.hidden_dim)
    grad[inputs:size(1)] = grad_outputs[1]
    grad_b[1] = grad_outputs[2]
  else
    grad   = torch.zeros(inputs:size(1), self.num_layers, self.hidden_dim)
    grad_b = torch.zeros(inputs:size(1), self.num_layers, self.hidden_dim)
    for l = 1, self.num_layers do
      grad[{inputs:size(1), l, {}}] = grad_outputs[1][l]
      grad_b[{1, l, {}}] = grad_outputs[2][l]
    end
  end
  self.rnn:backward(inputs, grad)
  return self.rnn_b:backward(inputs, grad_b, true)
end

function Encoder:parameters()
  return self.rnn:parameters()
end

function Encoder:forget()
  self.rnn:forget()
  if self.rnn_b then
    self.rnn_b:forget()
  end
end

return Encoder
