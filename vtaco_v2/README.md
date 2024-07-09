# ViTaM Algorithm

This repository mainly contains the deep-learning algorithm in the paper:

**Capturing forceful interaction with arbitrary objects using a deep learning-powered stretchable tactile array**

## Get-Started

The code is only tested on Ubuntu, we will soon test it on Windows system. 

## With conda and pip

Install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Supposing that the name `vtaco` is used for conda environment:

```shell
conda create -y -n vitam python=3.9
conda activate vitam
```

Then, install dependencies with `pip install`

```shell
pip install -r requirements.txt
```

In order to use the `manopth` module, go to the directory and install it

```shell
cd components/manopth
pip install .
```

## Dataset
We have prepared a small data sample in the directory `data`, with an object in the **bottle** category and a sequence of a hand grabing it. We will release the whole dataset sooner.

### Training
To train on this small data sample, simply use the following command

```shell
python train.py --config-name test
```

It will create a directory `output` which will generate the output of the training process, including tensorboard log files, visualization results, etc.