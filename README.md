# FlowColorization

Dataset has to be [donwloaded](https://davischallenge.org/davis2017/code.html) separately.

## ToDo's:
- Data:
  - finish optical flow implementation
- Network:
  - UNet, VAE, GAN ?
  - implement training checkpoints & weight saving
  - w/ and w/o optical flow

Layout I&O:
![NN IO Design](https://github.com/jan-spr/FlowColorization/blob/main/NN%20Diagram.png?raw=true)

## Requirements (so far):
- opencv-python
- scikit-image
- pytorch
- torchvision
- matplotlib
