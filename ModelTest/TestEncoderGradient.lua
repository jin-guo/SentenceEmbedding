require('..')
package.path = package.path .. ";./ModelTest/CheckGradient.lua"
require('CheckGradient')

function create_encoder(i_dim, h_dim, n_layers, rnn_type)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  local vocab_size = 500
  local emb_dim = i_dim
  local vecs = torch.Tensor(vocab_size, emb_dim)
  local encoder_config = {
    hidden_dim = h_dim,
    num_layers = n_layers,
    emb_vecs   = vecs,
    structure  = rnn_type
  }
  model = sentenceembedding.Encoder(encoder_config)

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.MSECriterion()
  return model, criterion
end

function fakedata(t, i_dim, hidden_dim, n_layers, rnn_type)
    local data = {}
    data.inputs = torch.rand(t,i_dim)
    if string.starts(rnn_type,'bi') then
      data.targets = torch.rand(2*n_layers*hidden_dim)
    else
      data.targets = torch.rand(n_layers*hidden_dim)
    end
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
torch.manualSeed(1)
local i_dim = 5
local h_dim = 2
local t = 10
local n_layers = 2
local rnn_type = 'bigru'
local data = fakedata(t, i_dim, h_dim, n_layers,rnn_type)
local model, criterion = create_encoder(i_dim, h_dim, n_layers, rnn_type)
local parameters, gradParameters = model:getParameters()


local f_encoder = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  local loss
  local outputs = model:forward(data.inputs)
  if model.num_layers>1 then
    if string.starts(model.structure,'bi') then
      outputs_forward = outputs[1]
      outputs_backward = outputs[2]
      local output_for_criterion_forward = {}
      local output_for_criterion_backward = {}
      for l = 1, n_layers do
        if l ==1 then
          output_for_criterion_forward = outputs_forward[l]
          output_for_criterion_backward = outputs_backward[l]
        else
          output_for_criterion_forward = torch.cat(output_for_criterion_forward,
            outputs_forward[l], 1)
          output_for_criterion_backward = torch.cat(output_for_criterion_backward,
            outputs_backward[l], 1)
        end
      end
      output_for_criterion = torch.cat(output_for_criterion_forward, output_for_criterion_backward, 1)
    else -- Case: only forward direction
      for l = 1, n_layers do
        output_for_current_layer = outputs[l]
        if l ==1 then
          output_for_criterion = output_for_current_layer
        else
          output_for_criterion = torch.cat(output_for_criterion,output_for_current_layer, 1)
        end
      end
    end
    loss  =  criterion:forward(output_for_criterion, data.targets)
  else -- Case: model.num_layers == 1
    if string.starts(model.structure,'bi') then
      outputs = torch.cat(outputs[1], outputs[2], 1)
    end
    loss = criterion:forward(outputs, data.targets)
  end
  -- Important: to clear the grad_input from the last forward step.
  model:forget()
  return loss
end

local g_encoder = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local loss, output_for_criterion, gradInput_cri
  local outputs = model:forward(data.inputs)
  if model.num_layers>1 then
    if string.starts(model.structure,'bi') then
      outputs_forward = outputs[1]
      outputs_backward = outputs[2]
      local output_for_criterion_forward = {}
      local output_for_criterion_backward = {}
      for l = 1, n_layers do
        if l ==1 then
          output_for_criterion_forward = outputs_forward[l]
          output_for_criterion_backward = outputs_backward[l]
        else
          output_for_criterion_forward = torch.cat(output_for_criterion_forward,
            outputs_forward[l], 1)
          output_for_criterion_backward = torch.cat(output_for_criterion_backward,
            outputs_backward[l], 1)
        end
      end
      output_for_criterion = torch.cat(output_for_criterion_forward, output_for_criterion_backward, 1)
    else -- Case: only forward direction
      for l = 1, n_layers do
        output_for_current_layer = outputs[l]
        if l ==1 then
          output_for_criterion = output_for_current_layer
        else
          output_for_criterion = torch.cat(output_for_criterion,output_for_current_layer, 1)
        end
      end
    end
    loss  =  criterion:forward(output_for_criterion, data.targets)
    gradInput_cri = criterion:backward(output_for_criterion, data.targets)
  else -- Case: model.num_layers == 1
    if string.starts(model.structure,'bi') then
      outputs = torch.cat(outputs[1], outputs[2], 1)
    end
    loss = criterion:forward(outputs, data.targets)
    gradInput_cri = criterion:backward(outputs, data.targets)
  end
  if model.num_layers>1 then
    local grad_for_encoder = {}
    local grad_for_encoder_forward = {}
    local grad_for_encoder_backward = {}

    for l = 1, model.num_layers do
      if string.starts(model.structure,'bi') then
        grad_for_encoder_forward[l] = gradInput_cri:narrow(1,
          1 + (l-1)*model.hidden_dim, model.hidden_dim)
        grad_for_encoder_backward[l] = gradInput_cri:narrow(1,
          1 + (l-1)*model.hidden_dim + model.num_layers*model.hidden_dim, model.hidden_dim)
      else
        grad_for_encoder[l] = gradInput_cri:narrow(1, 1 + (l-1)*model.hidden_dim, model.hidden_dim)
      end
    end
    if string.starts(model.structure,'bi') then
      grad_for_encoder = {grad_for_encoder_forward, grad_for_encoder_backward}
    end  -- Finished assemble grad_for_encoder
    model:backward(data.inputs, grad_for_encoder)

  else -- Case: model.num_layers == 1
    if string.starts(model.structure,'bi') then
      encoder_output_grads = {gradInput_cri:narrow(1, 1, model.hidden_dim),
        gradInput_cri:narrow(1, model.hidden_dim+1, model.hidden_dim)}
      model:backward(data.inputs, encoder_output_grads)
    else
      model:backward(data.inputs, gradInput_cri)
    end
  end
  return gradParameters
end

local diff = checkgrad(f_encoder, g_encoder, parameters)
print(diff)
