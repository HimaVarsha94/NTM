require('../')
require('./util')
require('optim')
require('sys')

torch.manualSeed(0)
obj = torch.load('repeat_copy.pkl','ascii')

local min_len = 1
local max_len = 20
local num_iters = 100
-- print(input_dim)
local input_dim = obj.input_dim

local start_symbol = torch.zeros(input_dim)
start_symbol[1] = 1
local end_symbol = torch.zeros(input_dim)

function generate_repeat_number(min, max)
    local k = (torch.rand(1)*max):ceil()
    return k[1]
end

function make_target(seq,k)
    local rows = seq:size()[1]
    local columns = seq:size()[2]
    local target = torch.zeros((k*rows)+1, columns)
    for i=1, rows*k,rows do
        target[{{i,i+rows-1},{}}] = seq
    end
    local end_limiter = torch.zeros(columns)
    end_limiter[2] = 1
    target[(k*rows)+1] = end_limiter
    return  target
end

function generate_sequence(len, bits)
  local seq = torch.zeros(len, bits + 2)
  for i = 1, len do
    seq[{i, {3, bits + 2}}] = torch.rand(bits):round()
  end
  return seq
end

function forward(model, seq,k, targets)
  local len = seq:size(1)
  local loss = 0

  -- present start symbol
  model:forward(start_symbol)

  -- present inputs
  for j = 1, len do
    model:forward(seq[j])
  end

  -- present end symbol
  model:forward(end_symbol)

  -- present targets
  local zeros = torch.zeros(input_dim)

  local outputs = torch.Tensor((k*len)+1, input_dim)
  local criteria = {}
  if print_flag then print('read head max') end

  for j = 1, k*len + 1 do
    criteria[j] = nn.BCECriterion()
    outputs[j] = model:forward(zeros)
    loss = loss + criteria[j]:forward(outputs[j], targets[j]) * input_dim
  end
  return outputs, criteria, loss
end


-- test
local test = torch.load('repeat_copy_testData.dat', 'ascii')
local input_seqs = test[1]
local ks = test[2]
local targets_table = test[3]
print(#input_seqs)
print(#ks)
print(#targets_table)

-- local inputs = {}
local losses = {}
-- local targets = {}
local outputs = {}
-- local criteria
for i = 1, num_iters do

    -- local len = math.floor(torch.random(min_len, max_len))
    local seq = input_seqs[i]
    local k = ks[i]
    -- inputs[i] = seq
    end_symbol[2] = k
    local targets = targets_table[i]
    outputs[i], criteria, losses[i] = forward(obj, seq,k, targets)

    -- return loss, grads

end
local out = {num_iters,output,losses}
torch.save('repeat_copy_out.dat', out, 'ascii')
