-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
function checkgrad(f, g, x, eps)
  -- compute true gradient
  local grad = g(x)

  -- compute numeric approximations to gradient
  local eps = eps or 1e-7
  print(eps)
  local grad_est = torch.DoubleTensor(grad:size())
  for i = 1, grad:size(1) do
    x[i] = x[i] + eps
    local loss_a = f(x)
    x[i] = x[i] - 2*eps
    local loss_b = f(x)
    x[i] = x[i] + eps
    grad_est[i] = (loss_a-loss_b)/(2*eps)
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / math.max(torch.norm(grad), torch.norm(grad_est))
  return diff, grad, grad_est
end

return checkgrad
