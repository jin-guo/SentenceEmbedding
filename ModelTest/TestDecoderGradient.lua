require('..')
package.path = package.path .. ";./test/CheckGradient.lua"
require('CheckGradient')

function create_decoder(i_dim, h_dim, n_layers, encoder_out_dim_real)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  local decoder_config = {
    encoder_out_dim = encoder_out_dim_real,
    in_dim          = i_dim,
    hidden_dim      = h_dim,
    num_layers      = n_layers
  }

  model = sentenceembedding.GRUDecoder(decoder_config)
  local linear_model = nn.Linear(n_layers*h_dim, i_dim)
  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.MSECriterion()
  return model, criterion, linear_model
end


function fakedata(t, i_dim, encoder_out_dim_real)
    local data = {}
    local sentence = torch.rand(t,i_dim)
    data.inputs = sentence:sub(1, -2)
    data.targets = sentence:sub(2,-1)
    data.encoder_output = torch.rand(encoder_out_dim_real)
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
torch.manualSeed(1)
local i_dim = 5
local h_dim = 3
local t = 10
local n_layers = 4
local encoder_out_dim_real = 10
local data = fakedata(t, i_dim, encoder_out_dim_real)
local model, criterion, linear_model = create_decoder(i_dim, h_dim, n_layers, encoder_out_dim_real)
local modules = nn.Parallel()
  :add(model)
  :add(linear_model)
local parameters, gradParameters = modules:getParameters()

local f_decoder = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  local loss, outputs
  local decoder_outputs = model:forward(data.inputs, data.encoder_output)
  if model.num_layers>1 then
    local decoder_output_flatten = torch.zeros(data.inputs:size(1), model.num_layers*model.hidden_dim)
    for t = 1, data.inputs:size(1) do
      local output_current_time = decoder_outputs:select(1, t)
      output_current_time = output_current_time:view(output_current_time:nElement())
      decoder_output_flatten[t] = output_current_time
    end
    outputs = linear_model:forward(decoder_output_flatten)
  else -- Case: model.num_layers == 1
    outputs = linear_model:forward(decoder_outputs)
  end
  loss = criterion:forward(outputs, data.targets)
  -- Important: to clear the grad_input from the last forward step.
  model:forget()
  return loss
end


local g_decoder = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local loss, output_for_criterion, gradInput_cri
  local decoder_outputs = model:forward(data.inputs, data.encoder_output)
  if model.num_layers>1 then
    local decoder_output_flatten = torch.zeros(data.inputs:size(1), model.num_layers*model.hidden_dim)
    for t = 1, data.inputs:size(1) do
      local output_current_time = decoder_outputs:select(1, t)
      output_current_time = output_current_time:view(output_current_time:nElement())
      decoder_output_flatten[t] = output_current_time
    end
    local outputs = linear_model:forward(decoder_output_flatten)
    loss = criterion:forward(outputs, data.targets)
    gradInput_cri = linear_model:backward(decoder_output_flatten, criterion:backward(outputs, data.targets))

  else -- Case: model.num_layers == 1
    local outputs = linear_model:forward(decoder_outputs)
    loss = criterion:forward(outputs, data.targets)
    gradInput_cri = linear_model:backward(decoder_outputs, criterion:backward(outputs, data.targets))
  end

  if model.num_layers>1 then
    local gradInput_for_decoder = torch.zeros(data.inputs:size(1), model.num_layers, model.hidden_dim)
    for t = 1, data.inputs:size(1) do
      gradInput_current_time = gradInput_cri:select(1,t)
      gradInput_current_time = gradInput_current_time:resize(model.num_layers, model.hidden_dim)
      gradInput_for_decoder[t] = gradInput_current_time
    end
    model:backward(data.inputs, gradInput_for_decoder)
  else -- Case: model.num_layers == 1
    model:backward(data.inputs, gradInput_cri)
  end
  return gradParameters
end

local diff = checkgrad(f_decoder, g_decoder, parameters)
print(diff)
