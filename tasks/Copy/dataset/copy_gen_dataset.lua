require('../../../')
require('../../util')
require('optim')
require('sys')

torch.manualSeed(0)

-- train gen config
local train_config = {
  input_dim = 10,
  output_dim = 10,
  num_items = 10000,
  min_len = 1,
  max_len = 20
  }

local test_config = {
    input_dim = 10,
    output_dim = 10,
    num_items = 100,
    min_len = 20,
    max_len = 120
}


function generate_sequence(len, bits)
  local seq = torch.zeros(len, bits + 2)
  for i = 1, len do
    seq[{i, {3, bits + 2}}] = torch.rand(bits):round()
  end
  return seq
end

function gen_train(train_config)
    local input_dim = train_config.input_dim
    local start_symbol = torch.zeros(input_dim)
    start_symbol[1] = 1
    local end_symbol = torch.zeros(input_dim)
    end_symbol[2] = 1
    local train_data = {}
    local train_targets = {}
    for i = 1,train_config.num_items do
        local len = math.floor(torch.random(train_config.min_len, train_config.max_len))
        local seq = generate_sequence(len, input_dim - 2)
        train_data[i] = seq
        train_targets[i] = seq
    end
    train = {train_data}
    return train
end

function gen_train(test_config)
    local input_dim = test_config.input_dim
    local start_symbol = torch.zeros(input_dim)
    start_symbol[1] = 1
    local end_symbol = torch.zeros(input_dim)
    end_symbol[2] = 1
    local test_data = {}
    local test_targets = {}
    for i = 1,test_config.num_items do
        local len = math.floor(torch.random(test_config.min_len, test_config.max_len))
        local seq = generate_sequence(len, input_dim - 2)
        test_data[i] = seq
        test_targets[i] = seq
    end
    test = {test_data}
    return test
end

local train = gen_train(train_config)
local test  = gen_train(test_config)

torch.save('copy_trainData.dat',train, 'ascii')
torch.save('copy_testData.dat',test, 'ascii')
