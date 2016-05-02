require('../../../')
require('../../util')
require('optim')
require('sys')

torch.manualSeed(0)

local pklDirectory = "../pre-trained-models/"

local dataDirectory = "../dataset/"

local pklFilename = ""

local dataFilename = dataDirectory.."recall_testData.dat"

if option == "1" then
  pklFilename = "recall_lstm_20k.pkl"
elseif option == "2" then
  pklFilename = "recall_gru_20k.pkl"
end

local pklFile = pklDirectory..pklFilename

local model = torch.load(pklFile)


local input_dim = model.input_dim

-- delimiter symbol and query symbol
local delim_symbol = torch.CudaTensor(input_dim):fill(0)
delim_symbol[1] = 1
local query_symbol = torch.CudaTensor(input_dim):fill(0)
query_symbol[2] = 1


function forward(model, items, num_queries)
  local num_items = #items
  local item_len = items[1]:size(1)
  local loss = 0

  -- present items
  for i = 1, num_items do
    model:forward(delim_symbol:cuda())
    for j = 1, item_len do
      model:forward(items[i][j]:cuda())
    end
  end

  -- present queries
  local zeros = torch.CudaTensor(input_dim):fill(0)
  local outputs = {}
  local criteria = {}
  local query_indices = {}
  for i = 1, num_queries do
    criteria[i] = {}
    outputs[i] = torch.CudaTensor(item_len, input_dim)

    local query_idx = math.floor(torch.uniform(1, num_items))
    query_indices[i] = query_idx
    local query = items[query_idx]
    local target = items[query_idx + 1]

    -- query
    model:forward(query_symbol):cuda()

    for j = 1, item_len do
      model:forward(query[j]:cuda())
    end

    -- target
    model:forward(query_symbol:cuda())
    for j = 1, item_len do
      criteria[i][j] = nn.BCECriterion()
      outputs[i][j] = model:forward(zeros):cuda()
      loss = loss + criteria[i][j]:forward(outputs[i][j], target[j]:cuda()) * input_dim
    end
  end
  return query_indices, outputs, criteria, loss
end



local num_iters = 100
local train = torch.load(dataFilename, 'ascii')
local inputs = train[1]
local num_queries = 1
local outputs = {}
local losses = {}
local query_indices = {}

for iter = 1,(#inputs) do
    local items = inputs[iter]
    print('iter = ' .. iter)
    local query_index, output, criteria, sample_loss = forward(model, items, num_queries)
    outputs[iter] = output
    query_indices[iter] = query_index
    losses[iter] = sample_loss
    print("loss = "..sample_loss)
end
