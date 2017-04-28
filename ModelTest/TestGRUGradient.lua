require('..')
package.path = package.path .. ";./test/CheckGradient.lua"
require('CheckGradient')

--------------------------------------------------------------
-- SETTINGS
function create_gru(i_dim, h_dim, n_layers)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  local gru_config = {
    in_dim = i_dim,
    hidden_dim = h_dim,
    num_layers = n_layers,
    gate_output = true
  }
  model = sentenceembedding.GRU(gru_config)

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.MSECriterion()
  return model, criterion
end

function fakedata(t, i_dim, hidden_dim, n_layers)
    local data = {}
    data.inputs = torch.rand(t,i_dim)
    data.targets = torch.rand(n_layers*hidden_dim)
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
torch.manualSeed(1)
local i_dim = 5
local h_dim = 5
local t = 10
local n_layers = 4
local data = fakedata(t, i_dim, h_dim, n_layers)
local model, criterion = create_gru(i_dim, h_dim, n_layers)
local parameters, gradParameters = model:getParameters()

-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  local output = model:forward(data.inputs)
  local output_for_criterion, loss
  if model.num_layers>1 then
    output_for_criterion = output[1]
    for l = 2, n_layers do
       output_for_criterion = torch.cat(output_for_criterion,output[l],1)
    end
    loss  =  criterion:forward(output_for_criterion, data.targets)
  else
    loss =  criterion:forward(output, data.targets)
  end
  -- Important: to clear the grad_input from the last forward step.
  model:forget()
  return loss
end

-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local output = model:forward(data.inputs)
  local output_for_criterion, loss
  local gradInput_cri, grad
  if model.num_layers>1 then
    output_for_criterion = output[1]
    for l = 2, n_layers do
       output_for_criterion = torch.cat(output_for_criterion,output[l], 1)
    end
    loss  =  criterion:forward(output_for_criterion, data.targets)
    gradInput_cri = criterion:backward(output_for_criterion, data.targets)
    grad = torch.zeros(data.inputs:size(1), model.num_layers, model.hidden_dim)
    for l = 1, model.num_layers do
      grad[{data.inputs:size(1), l, {}}] =
        gradInput_cri:narrow(1, 1 + (l-1)*model.hidden_dim, model.hidden_dim)
    end
  else
    loss =  criterion:forward(output, data.targets)
    gradInput_cri = criterion:backward(output, data.targets)
    grad = torch.zeros(data.inputs:size(1), model.hidden_dim)
    grad[data.inputs:size(1)] = gradInput_cri
  end


  model:backward(data.inputs, grad)
  return gradParameters
end

local diff = checkgrad(f, g, parameters)
print(diff)
