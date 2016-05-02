require('../../../')
require('../../util')
require('optim')
require('sys')
-- require 'gnuplot'

local pklDirectory = "../pre-trained-models/"

local pklFilename = ""

local dataDirectory = "../dataset/"

local dataFilename = dataDirectory.."copy_testData.dat"

if option == "1" then
  pklFilename = "copy_lstm.pkl"
elseif option == "2" then
  pklFilename = "copy_gru.pkl"
end

local pklFile = pklDirectory..pklFilename
print(pklFile)

obj = torch.load(pklFile,'ascii')

-- local min_len = 1
-- local max_len = 20
local num_iters = 100
-- print(input_dim)
local input_dim = obj.input_dim

local start_symbol = torch.zeros(input_dim)
start_symbol[1] = 1
local end_symbol = torch.zeros(input_dim)




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
local test = torch.load(dataFilename, 'ascii')
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
    print('iter = ' .. i)
    print('loss = ' .. losses[i])


    -- return loss, grads

end
local out = {output,targets_table}
torch.save('../results/repeat_copy_out.dat', out, 'ascii')
