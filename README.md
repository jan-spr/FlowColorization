# FlowColorization

Dataset has to be [donwloaded](https://davischallenge.org/davis2017/code.html) separately.

## Contents:
1. [UNet_train_noFlow.ipynb](UNet_train_noFlow.ipynb)
NB for training the UNet Model
2. NB for training the UNet Model


## ToDo's:
- Data:
  - test/train split
  - data samples including >1 frame difference
- Network:
  - UNet, VAE, GAN ?
  - think about Loss function
- Results
  - show performance of fully trained Model (make NB)

Layout I&O:
![NN IO Design](https://github.com/jan-spr/FlowColorization/blob/main/NN%20Diagram.png?raw=true)

## Requirements (so far):
- opencv-python
- scikit-image
- pytorch
- torchvision
- matplotlib
