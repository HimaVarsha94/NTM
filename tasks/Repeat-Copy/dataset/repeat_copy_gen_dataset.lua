require('../../')
require('../util')
require('optim')
require('sys')

torch.manualSeed(0)

-- NTM config
local train_config = {
  input_dim = 10,
  output_dim = 10,
  min_len = 1,
  max_len = 20,
  max_k = 10,
  dataset_size = 5000
}

local test_config = {
  input_dim = 10,
  output_dim = 10,
  min_len = 1,
  max_len = 20,
  max_k = 20,
  dataset_size = 100
}


local input_dim = train_config.input_dim
-- local start_symbol = torch.zeros(input_dim)
-- start_symbol[1] = 1
-- local end_symbol = torch.zeros(input_dim)

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

function generate_data(config)
    local num_items = config.dataset_size
    local input_seq = {}
    local ks = {}
    local targets = {}
    for i = 1, num_items do
        local len = math.floor(torch.random(config.min_len, config.max_len))
        local seq = generate_sequence(len, config.input_dim - 2)
        local k = generate_repeat_number(1,config.max_k)
        -- end_symbol[2] = k
        input_seq[i] = seq
        ks[i] = k
        targets[i] = make_target(seq,k)
    end
    data = {input_seq, ks, targets}
    print(#data)
    print(#data[1])
    print(#data[2])
    print(#data[3])


    return data
end

train = generate_data(train_config)
test = generate_data(test_config)

torch.save('repeat_copy_trainData.dat',train, 'ascii')
torch.save('repeat_copy_testData.dat',test, 'ascii')
