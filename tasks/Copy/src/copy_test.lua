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

--print("Here")

local input_dim = obj.input_dim
-- local len = 8
-- local min_len = 20
-- local max_len = 120
local start_symbol = torch.zeros(input_dim)
start_symbol[1] = 1
local end_symbol = torch.zeros(input_dim)
end_symbol[2] = 1

-- function generate_sequence(len, bits)
--   local seq = torch.zeros(len, bits + 2)
--   for i = 1, len do
--     seq[{i, {3, bits + 2}}] = torch.rand(bits):round()
--   end
--   return seq
-- end

function forward(model, seq, print_flag)
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
  local outputs = torch.Tensor(len, input_dim)
  local criteria = {}
  for j = 1, len do
    criteria[j] = nn.BCECriterion()
    outputs[j] = model:forward(zeros)
    loss = loss + criteria[j]:forward(outputs[j], seq[j]) * input_dim
  end
  return outputs, criteria, loss
end

local inputs = {}
local losses = {}
local outputs = {}
local targets = {}

for i = 1, 100 do
    -- local seq = generate_sequence(len, (input_dim) - 2)
    local seq = inputs[i]
    targets[i] = seq
    outputs[i], criteria, losses[i] = forward(obj, seq,0)
    print(losses[i])
end

local out = {outputs,targets}
torch.save("../results/copy_test_out.dat", out,'ascii')

-- gnuplot.plot({torch.range(1,#losses),torch.Tensor(losses),'-'})
