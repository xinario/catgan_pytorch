# catGAN

PyTorch implementation of [Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks](https://arxiv.org/abs/1511.06390) that was originally proposed by Jost Tobias Springenberg.




### Results on CIFAR10
Note that in this repo, only the unsupervised version was implemented for now. I reaplced the orginal architecture with DCGAN and the results are more colorful than the original one.

From 0 to 100 epochs:

![cifar10](results/cifar10/cifar10.gif)



## Prerequisites
- Python 2.7
- PyTorch v0.2.0
- Numpy
- SciPy
- Matplotlib


## Getting Started
### Installation
- Install [PyTorh](https://github.com/pytorch/pytorch) and the other dependencies
- Clone this repo:
```bash
git clone https://github.com/xinario/catgan_pytorch.git
cd catgan_pytorch
```

### Train
- Download the cifar10 dataset and put it inside ./datasets/cifar10:

- Train a model:
```bash
python catgan_cifar10.py --data_dir ./datasets/cifar10 --name cifar10
```
All the generated plot and samples can be found in side ./results/cifar10




### Training options 
```bash
optional arguments:

--continue_train  	to continue training from the latest checkpoints if --netG and --netD are not specified
--netG NETG           path to netG (to continue training)
--netD NETD           path to netD (to continue training)
--workers WORKERS     number of data loading workers
--num_epochs EPOCHS         number of epochs to train for
```
More options can be found in side the training script.



## Acknowledgments
Some of code are inspired and borrowed from [wgan-gp](https://github.com/caogang/wgan-gp), [DCGAN](https://github.com/pytorch/examples/tree/master/dcgan), [catGAN chainer repo](https://github.com/smayru/catgan)
