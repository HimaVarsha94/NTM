require('../../../')
require('../../util')
require('optim')
require('sys')

torch.manualSeed(0)
-- print(option)

local pklDirectory = "../pre-trained-models/"

local pklFilename = ""

local dataDirectory = "../dataset/"

local dataFilename = dataDirectory.."copy_trainData.dat"

if option == "1" then
  pklFilename = "copy_lstm.pkl"
elseif option == "2" then
  pklFilename = "copy_gru.pkl"
end

local pklFile = pklDirectory..pklFilename
print(pklFile)

-- NTM config
local config = {
  input_dim = 10,
  output_dim = 10,
  mem_rows = 128,
  mem_cols = 20,
  cont_dim = 100
}

local input_dim = config.input_dim
local start_symbol = torch.CudaTensor(input_dim):fill(0)
start_symbol[1] = 1
local end_symbol = torch.CudaTensor(input_dim):fill(0)
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
  for j = 1, len do
    model:forward(seq[j]:cuda())
  end

  -- present end symbol
  model:forward(end_symbol)

  -- present targets
  local zeros = torch.CudaTensor(input_dim):fill(0)
  local outputs = torch.CudaTensor(len, input_dim)
  local criteria = {}
  for j = 1, len do
    criteria[j] = nn.BCECriterion()
    outputs[j] = model:forward(zeros)
    loss = loss + criteria[j]:forward(outputs[j], seq[j]) * input_dim
  end
  return outputs, criteria, loss
end

function backward(model, seq, outputs, criteria)
  local len = seq:size(1)
  local zeros = torch.CudaTensor(input_dim):fill(0)
  for j = len, 1, -1 do
      criteria[j] = criteria[j]:cuda()
    model:backward(
      zeros,
      criteria[j]
        :backward(outputs[j], seq[j])
        :mul(input_dim)
      )
  end

  model:backward(end_symbol, zeros)
  for j = len, 1, -1 do
    model:backward(seq[j], zeros)
  end
  model:backward(start_symbol, zeros)
end

local model = ntm.NTM(config):cuda()
local params, grads = model:getParameters()
params = params:cuda()
grads = grads:cuda()

local num_iters = 10000
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

train = torch.load(dataFilename, 'ascii')

inputs = train[1]


-- train
for iter = 1, num_iters do
  local print_flag = (iter % print_interval == 0)
  local feval = function(x)
    if print_flag then
     --  print(string.rep('-', 80))
      print('iter = ' .. iter)
     --  print('learn rate = ' .. rmsprop_state.learningRate)
      -- print('momentum = ' .. rmsprop_state.momentum)
      -- print('decay = ' .. rmsprop_state.decay)
      -- printf('t = %.1fs\n', sys.clock() - start)
    end

    local loss = 0
    grads:zero()

    
    local seq = inputs[iter]
    seq = seq:cuda()
    local outputs, criteria, sample_loss = forward(model, seq, print_flag)
    loss = loss + sample_loss
    backward(model, seq, outputs, criteria)
    if print_flag then
      -- print("target:")
      -- print(seq)
      -- print("output:")
      -- print(outputs)
    end

    -- clip gradients
    grads:clamp(-10, 10)
    if print_flag then
      -- print('max grad = ' .. grads:max())
      -- print('min grad = ' .. grads:min())
      print('loss = ' .. loss)
    end
    return loss, grads
  end

  --optim.adagrad(feval, params, adagrad_state)
  ntm.rmsprop(feval, params, rmsprop_state)
end
torch.save(pklFile, model);
