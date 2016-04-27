--[[

  Training a NTM to memorize input.

  The current version seems to work, giving good output after 5000 iterations
  or so. Proper initialization of the read/write weights seems to be crucial
  here.

--]]

require('../')
require('./util')
require('optim')
require('sys')

torch.manualSeed(0)

-- NTM config
local config = {
  input_dim = 10,
  output_dim = 10,
  mem_rows = 128,
  mem_cols = 20,
  cont_dim = 100
}


local input_dim = config.input_dim
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

function forward(model, seq, print_flag,k, targets)
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

  local outputs = torch.Tensor((k*len)+1, input_dim)
  local criteria = {}
  if print_flag then print('read head max') end

  for j = 1, k*len + 1 do
    criteria[j] = nn.BCECriterion()
    outputs[j] = model:forward(zeros)
    loss = loss + criteria[j]:forward(outputs[j], targets[j]) * input_dim
    if print_flag then print_read_max(model) end
  end
  return outputs, criteria, loss
end

function backward(model, seq, outputs, criteria, k, targets)
  local len = seq:size(1)
  local zeros = torch.zeros(input_dim)
  for j = k*len+1, 1, -1 do
    model:backward(
      zeros,
      criteria[j]
        :backward(outputs[j], targets[j])
        :mul(input_dim)
      )
  end

  model:backward(end_symbol, zeros)
  for j = len, 1, -1 do
    model:backward(seq[j], zeros)
  end
  model:backward(start_symbol, zeros)
end

local model = ntm.NTM(config)
local params, grads = model:getParameters()

local num_iters = 10
local start = sys.clock()
local print_interval = 50
local min_len = 1
local max_len = 20

print(string.rep('=', 80))
print("NTM copy task")
print('training up to ' .. num_iters .. ' iteration(s)')
print('min sequence length = ' .. min_len)
print('max sequence length = ' .. max_len)
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local rmsprop_state = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

-- local adagrad_state = {
--   learningRate = 1e-3
-- }
local train = torch.load('repeat_copy_trainData.dat', 'ascii')
local input_seqs = train[1]
local ks = train[2]
local targets_table = train[3]
print(#input_seqs)
print(#ks)
print(#targets_table)


-- train
for iter = 1, num_iters do
  local print_flag = (iter % print_interval == 0)
  local feval = function(x)
    if print_flag then
      print(string.rep('-', 80))
      print('iter = ' .. iter)
      print('learn rate = ' .. rmsprop_state.learningRate)
      print('momentum = ' .. rmsprop_state.momentum)
      print('decay = ' .. rmsprop_state.decay)
      printf('t = %.1fs\n', sys.clock() - start)
    end

    local loss = 0
    grads:zero()
    -- local len = math.floor(torch.random(min_len, max_len))
    local seq = input_seqs[iter]
    -- print(seq)
    local k = ks[iter]
    -- print(k)
    end_symbol[2] = k
    local targets = targets_table[iter]
    -- print(targets)

    local outputs, criteria, sample_loss = forward(model, seq, print_flag,k, targets)
    loss = loss + sample_loss
    backward(model, seq, outputs, criteria,k, targets)
    if print_flag then
      print("target:")
      print(seq)
      print("output:")
      print(outputs)
    end

    -- clip gradients
    grads:clamp(-10, 10)
    if print_flag then
      print('max grad = ' .. grads:max())
      print('min grad = ' .. grads:min())
      print('loss = ' .. loss)
    end
    return loss, grads
  end

  --optim.adagrad(feval, params, adagrad_state)
  ntm.rmsprop(feval, params, rmsprop_state)
end

torch.save("repeat_copy.pkl", model,'ascii');
