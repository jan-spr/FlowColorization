# FlowColorization

Dataset has to be [donwloaded](https://davischallenge.org/davis2017/code.html) separately.

## ToDo's:
- Data:
  - finish optical flow implementation
  - test/train split
- Network:
  - UNet, VAE, GAN ?
  - implement training checkpoints & weight saving
  - w/ and w/o optical flow
  - think about Loss function
- NN training
  - save NN state & loss trajectory
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
