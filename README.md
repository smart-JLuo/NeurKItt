# Neural Krylov Iteration for Accelerating Linear System Solving

This is the official implementation of the paper "Neural Krylov Iteration for Accelerating Linear System Solving" in NeurIPS 2024.

## File Tree
- data `(Save the dataset in this folder.)`
  - helmholtz.py 
- README.md
- requirements.txt
- model.py `(Where the model)`
- loss.py `(Where the loss function)`
- data_utils.py `(Tools helping dataloading)`
- solve.py `(Where we solve the linear system problem)`
- train.py `(Training and saving the subspace prediction model)`
- e.c `(C++ file for solving. Compiling it before running.)`
- makefile

## Setup

**WARNING**: Before you install the dependecies below, you are supposed to have installed PETSc (C++ version 3.19.4) and petsc4py following the instruction from [PETSc document](https://petsc.org/release/install/). To run the `solve.py`, you should compile `e.c` first.

You can find other dependencies in `requirements.txt`. A script for installation is shown as follows:

```shell
conda create -n neurkitt python=3.10
conda activate neurkitt

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.25.2
pip install scipy====1.11.1
pip install tqdm json
```

## Train and Reproduction

To run NeurKItt, you can use the following command.
```shell
python train.py
```
We set Helmholtz for default. You can also use your own model and data. 

After training the subspace prediction module, you will have the predicted subspace, then running the `solve.py` to solve your linear systems.
```shell
python solve.py
```

If you find our work useful your research, please cite our paper:

```
@inproceedings{luoneural,
  title={Neural Krylov Iteration for Accelerating Linear System Solving},
  author={Luo, Jian and Wang, Jie and Wang, Hong and Geng, Zijie and Chen, Hanzhu and Kuang, Yufei and others},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
