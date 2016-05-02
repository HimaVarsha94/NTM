require('../../../')
require('../../util')
require('optim')
require('sys')

torch.manualSeed(0)

local pklDirectory = "../pre-trained-models/"

local dataDirectory = "../dataset/"

local pklFilename = ""

local dataFilename = dataDirectory.."recall_trainData.dat"

if option == "1" then
  pklFilename = "recall_lstm.pkl"
elseif option == "2" then
  pklFilename = "recall_gru.pkl"
end

local pklFile = pklDirectory..pklFilename
print(pklFile)
local config = {
  input_dim = 8,
  output_dim = 8,
  mem_rows = 128,
  mem_cols = 20,
  cont_dim = 100
}


local input_dim = config.input_dim

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

function backward(model, items, query_indices, outputs, criteria)
  local num_queries = #query_indices
  local num_items = #items
  local item_len = items[1]:size(1)
  local zeros = torch.CudaTensor(input_dim):cuda()
  for i = num_queries, 1, -1 do
    local query_idx = query_indices[i]
    local query = items[query_idx]
    local target = items[query_idx + 1]:cuda()

    -- target
    for j = item_len, 1, -1 do
      criteria[i][j] = criteria[i][j]:cuda()
      model:backward(
        zeros, criteria[i][j]:backward(outputs[i][j], target[j]:cuda()):mul(input_dim))
    end
    model:backward(query_symbol:cuda(), zeros)

    -- query
    for j = item_len, 1, -1 do
      model:backward(query[j]:cuda(), zeros)
    end
    model:backward(query_symbol:cuda(), zeros)
  end

  for i = num_items, 1, -1 do
    local item = items[i]
    for j = item_len, 1, -1 do
      model:backward(item[j]:cuda(), zeros)
    end
    model:backward(delim_symbol:cuda(), zeros)
  end
end


local model = ntm.NTM(config):cuda()


local params, grads = model:getParameters()
params = params:cuda()
grads = grads:cuda()
local num_iters = 50000
local min_len = 2
local max_len = 6
local item_len = 3
print(string.rep('=', 80))
print("NTM associative recall task")
print('training up to ' .. num_iters .. ' iteration(s)')
print('min sequence length = ' .. min_len)
print('max sequence length = ' .. max_len)
print('sequence element length = ' .. item_len)
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local rmsprop_state = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

train = torch.load(dataFilename, 'ascii')
inputs = train[1]
local losses = {}
-- train
local start = sys.clock()
local print_interval = 50
for iter = 1, num_iters do
  local print_flag = (iter % print_interval == 0)
  local num_queries = 1
  local items = inputs[iter]

  local feval = function(x)
    if print_flag then
    --   print(string.rep('=', 80))
      print('iter = ' .. iter)
    end

    local loss = 0
    grads:zero()

    local query_indices, outputs, criteria, sample_loss = forward(
      model, items, num_queries)

    loss = loss + sample_loss
    backward(model, items, query_indices, outputs, criteria)

    -- clip gradients
    grads:clamp(-10, 10)
    if print_flag then
      print('loss = ' .. loss)
      losses[#losses + 1] = loss
    end
    return loss, grads
  end
    if iter == 5000 then
      if option == "1" then

        torch.save(pklDirectory..'recall_lstm_5k.pkl', model)
      elseif option == "2" then
        torch.save(pklDirectory..'recall_gru_5k.pkl', model)
      end
    end
    if iter == 10000 then
      if option == "1" then
        torch.save(pklDirectory..'recall_lstm_10k.pkl', model)
      elseif option == "2" then
        torch.save(pklDirectory..'recall_gru_10k.pkl', model)
      end
    end
    if iter == 15000 then
      if option == "1" then
        torch.save(pklDirectory..'recall_lstm_15k.pkl', model)
      elseif option == "2" then
        torch.save(pklDirectory..'recall_gru_15k.pkl', model)
      end
    end

    if iter == 20000 then
    	if option == "1" then
        torch.save(pklDirectory..'recall_lstm_20k.pkl', model)
      elseif option == "2" then
        torch.save(pklDirectory..'recall_gru_20k.pkl', model)
      end
    end
    if iter == 40000 then
    	if option == "1" then
        torch.save(pklDirectory..'recall_lstm_40k.pkl', model)
      elseif option == "2" then
        torch.save(pklDirectory..'recall_gru_40k.pkl', model)
      end
    end

  ntm.rmsprop(feval, params, rmsprop_state)
end
-- torch.save('recall_lstm_trainLoss.dat', losses, 'ascii')

torch.save(pklFile, model)
