require('../')
require('./util')
require('optim')
require('sys')

obj = torch.load('copy.pkl')
local input_dim = obj.input_dim
local len = 8
local min_len = 1
local max_len = 20
local start_symbol = torch.zeros(input_dim)
start_symbol[1] = 1
local end_symbol = torch.zeros(input_dim)
end_symbol[2] = 1

function generate_sequence(len, bits)
  local seq = torch.zeros(len, bits + 2)
  for i = 1, len do
    seq[{i, {3, bits + 2}}] = torch.rand(bits):round()
  end
  return seq
end

function forward(model, seq, print_flag)
  local len = seq:size(1)
  local loss = 0

  -- present start symbol
  model:forward(start_symbol)

  -- present inputs
  if print_flag then print('write head max') end
  for j = 1, len do
    model:forward(seq[j])
    if print_flag then print_write_max(model) end
  end

  -- present end symbol
  model:forward(end_symbol)

  -- present targets
  local zeros = torch.zeros(input_dim)
  local outputs = torch.Tensor(len, input_dim)
  local criteria = {}
  if print_flag then print('read head max') end
  for j = 1, len do
    criteria[j] = nn.BCECriterion()
    outputs[j] = model:forward(zeros)
    loss = loss + criteria[j]:forward(outputs[j], seq[j]) * input_dim
    if print_flag then print_read_max(model) end
  end
  return outputs, criteria, loss
end

local seq = generate_sequence(len, (input_dim) - 2)
output, criteria, loss = forward(obj, seq,1)
print(loss)
print(output)
