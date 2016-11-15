A Neural Turing Machine in Torch
================================

A Torch implementation of the Neural Turing Machine model described in this
[paper](http://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne and Ivo Danihelka.

This implementation uses an LSTM and a GRU controller. Thanks to [kaishengtai](https://github.com/kaishengtai)'s [implementation](https://github.com/kaishengtai/torch-ntm) from which this is built on. NTM models with multiple read/write heads are supported. Also the implementation has both cuda and non-cuda versions. 

## Requirements

[Torch7](https://github.com/torch/torch7), as well as the following
libraries:

[nn](https://github.com/torch/nn)

[optim](https://github.com/torch/optim)

[nngraph](https://github.com/torch/nngraph)

[cutorch](https://github.com/torch/cutorch)

[cunn](https://github.com/torch/cunn)

All the above dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

## Project Structure
The base directory contains the implementation of NTM with LSTM and GRU controllers. The layers directory contains the implementation of some modules which is used in the NTM.

The task directory contains the data, train and test code, and pre-trained models and results. There are three tasks namely Copy, Repeat-Copy and Recall each in their respective directory. For every task there are four directories named

1. dataset (contains the script to generate data for test and train and already generated data files)

2. src (contains script for training and testing)

3. pre-trained-models (contains trained models in pkl files)

4. results (contains results for train and test)

To generate data go to dataset and type:
```
$ th <task>_gen_dataset.lua
```
It will generate two data set one for train and one for test.

To train go to src/ and type:
```
$ th <task>_cuda.lua
```
To test go to src/ and type:
```
$ th <task>_test.lua
```

The pre-trained/ directory contains pre-trained pkl models which can be used for testing.


## Usage

There are multiple tasks which this repo implements. The tasks implemented are:
1. Copy
2. Repeat Copy
3. Associative Recall

To check a particular task, for example Recall
1. Generate dataset
```
$ cd tasks/Recall
$ th recall_gen_dataset.lua
```
2. Train the model  
```
$ cd ../src
$ th recal_cuda.lua
```
The program will ask the architecture. Give 1 for LSTM and 2 for GRU

3. Test our model
```
$ th recall_test.lua
```

## References:

1. [Kaishengtai/torch-ntm Github Repository](https://github.com/kaishengtai/torch-ntm)
2. [Neural Turing Machines ](http://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne and Ivo Danihelka
