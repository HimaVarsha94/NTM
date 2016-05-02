require('../../')
require('../util')
require('optim')
require('sys')

torch.manualSeed(0)
local train_config = {
  input_dim = 8,
  output_dim = 8,
  min_len = 2,
  max_len = 6,
  item_len = 3,
  dataset_size = 15000
}

local test_config = {
  input_dim = 8,
  output_dim = 8,
  min_len = 2,
  max_len = 6,
  item_len = 3,
  dataset_size = 100
}

local input_dim = train_config.input_dim
function generate_items(num_items, item_len)
  local items = {}
  print(num_items)

  for i = 1, num_items do
    local item = torch.rand(item_len, input_dim):round()
    for j = 1, item_len do
      item[{j, {1, 2}}]:zero()
    end
    table.insert(items, item)
  end
  return items
end

function generate_data(config)
    local item_size = config.dataset_size
    local input_seq = {}
    for i = 1, item_size do
        local num_items = math.floor(torch.uniform(config.min_len, config.max_len + 1))
        local num_queries = math.floor(torch.uniform(1, num_items + 1))
        local items = generate_items(num_items, config.item_len)
        input_seq[i] = items
    end
    data = {input_seq}
    return data
end

train = generate_data(train_config)
test = generate_data(test_config)

torch.save('recall_trainData.dat',train, 'ascii')
torch.save('recall_testData.dat',test, 'ascii')
