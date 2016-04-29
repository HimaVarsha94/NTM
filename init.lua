require('torch')
require('nn')
require('nngraph')

ntm = {}
printf = utils.printf
local option = "1"
repeat
   io.write("Select a Variant of NTM to run:  1. NTM_LSTM 2. NTM_LSTM_cuda 3. NTM_GRU 4. NTM_GRU_cuda ")
   io.flush()
   option=io.read()
until option=="1" or option=="2" or option=="3" or option=="4"

if option == "2" or option == "4" then
    require('cunn')
    require('cutorch')
    cutorch.setDevice(4);
end
include('rmsprop.lua')
include('layers/CircularConvolution.lua')
include('layers/OuterProd.lua')
include('layers/PowTable.lua')
include('layers/Print.lua')
include('layers/SmoothCosineSimilarity.lua')
include('layers/ScalarMulTable.lua')
include('layers/ScalarDivTable.lua')
if option == "1" then
  include('ntm_lstm.lua')
elseif option == "2" then
    include('ntm_lstm_cuda.lua')
elseif option == "3" then
  include('ntm_gru.lua')
elseif option == "4" then
  include('ntm_gru_cuda.lua')
end

function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end
