import torch 


# suppose model is defined and loaded  

# the model is evaluated at x, gradient is taken at x as well
# u  = grad (model (x) ) (v)

device = torch.device ( 'cuda' )
dtype =  torch.float32

for t, (x, y) in enumerate(test_dataloader):
# x: network input, y label

  x.requires_grad =True
  prediction = model(x)

# dummy can be anything, like a placeholder
  dummy_pt =  torch.from_numpy (dummy).to (device = device, dtype =dtype)
  dummy_pt.requires_grad = True

# backpropagating dummy to the input (x, to obtain \bar{x})  ---> vjp
  vjp = torch.autograd.grad (prediction, x, grad_outputs = dummy_pt, create_graph  = True ) [0]


  v_pt =  torch.from_numpy (v).to (device = device, dtype =dtype)

# now treating the mapping between dummy_pt --> vjp as the forward map (linear in dummy_pt), and backpropagate
  u = torch.autograd.grad (vjp, dummy_pt, grad_outputs  = v_pt) [0]
