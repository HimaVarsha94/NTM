A Neural Turing Machine in Torch
================================

A Torch implementation of the Neural Turing Machine model described in this
[paper](http://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne and Ivo Danihelka.

This implementation uses an LSTM/GRU controller. NTM models with multiple read/write heads are supported. Also the implementation has both cuda and non-cuda versions.

## Requirements

[Torch7](https://github.com/torch/torch7), as well as the following
libraries:

[nn](https://github.com/torch/nn)

[optim](https://github.com/torch/optim)

[nngraph](https://github.com/torch/nngraph)

If you are using cuda code then the following libraries as well:

[cutorch](https://github.com/torch/cutorch)

[cunn](https://github.com/torch/cunn)

All the above dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

## Usage

There are multiple tasks which this repo implements. The tasks implemented are:
1. Copy
2. Repeat Copy
3. Associative Recall

To implement a particular task, for example repeat_copy task
1. Generate dataset
```
cd tasks
th tasks/repeat_copy_gen_dataset.lua
```
2. Train the model
```
th tasks/repeat_copy.lua
```
Then choose the architecture to be used by entering 1,2,3 or 4.

3. Test our model
```
th tasks/repeat_copy_test.lua
```

## References:

1. [Kaishengtai/torch-ntm Github Repository](https://github.com/kaishengtai/torch-ntm)
2. [Neural Turing Machines ](http://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne and Ivo Danihelka
