# Image Recognition Bus Type Classification

This deep learning CNN model was built using the fastai ai library, which sits on top of Pytorch v1. The GPU used was an NVIDIA P4 GPU with 8 CPU's and 52 GB of RAM run from the Google Cloud Platform (GCP). The network architecture used was resnet-34, which is a CNN with pre-trained weights from the ImageNet library. 

This is an image classifier model that can tell the difference between 2 types of buses used in Panama city. The classic type called *Diablo Rojo* that is being phased out. And the modern type called *Metrobus*. The accuracy of the model on a validation set was 98.2%. As you can see it’s pretty easy to tell the difference.

<img src="https://github.com/mlsmall/Image-Recognition-Bus-Type-Classification/blob/master/bus%20types.png" width="450"\>
