# Building detection

This section explains how the building detection works. We base our work on building recognition by training a neural network architecture using deep learning with the open source Pytorch lightning library.

## 1. Training

### Datas

The training is based on open source data from the xView2 challenge : https://www.xview2.org/. 
These data are MAXAR satellite images applied to observations of natural disasters around the world. They are composed of (1024,1024) RGB images and building annotation polygons. This data makes it easy to train a neural network based on deep learning with Pytorch lightning. We use semantic segmentation, which ensures that the mask contains only one binary class (1 if the pixel is a building in the image and 0 otherwise). Once the images and masks have been defined in matrices, we need to define the neural network parameters.

### Neural network

## 2. Inference