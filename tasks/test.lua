require "torch"
require "cutorch"
local train = torch.load('copy_train.dat', 'ascii')
-- train = torch.Tensor(train)
local data = train[1]
local targets = train[2]

print(#data)
print(#targets)
