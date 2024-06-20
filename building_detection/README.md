# Building detection

This section explains how the building detection works. We base our work on building recognition by training a neural network architecture using deep learning with the open source Pytorch lightning library.

## 1. Training

### Datas

The training is based on open source data from the xView2 challenge : https://www.xview2.org/. 
These data are MAXAR satellite images applied to observations of natural disasters around the world. They are composed of (1024,1024) RGB images and building annotation polygons. This data makes it easy to train a neural network based on deep learning with Pytorch lightning. We use semantic segmentation, which ensures that the mask contains only one binary class (1 if the pixel is a building in the image and 0 otherwise). Once the images and masks have been defined in matrices, we need to define the neural network parameters.

### Neural network

Before training your neural network, you need to define the right parameters to avoid overlearning, underlearning, convergence problems and poor overall model performance. That's why you can change the drive parameters by: selecting the architecture, encoder, network depth and output activation function.
```
my_model = get_model(
    num_classes = 1,                        # For semantic segmentation, we want just one class
    neural_network = 'name_of_my_NN',       # UNet, DeepLabV3, MAnet, Linknet, FPN, PAN, ...
    encoder_name = 'name_of_my_encoder',    # ResNet, GERNet, DenseNet, EfficientNet, MobileNet, ...
    encoder_depth = depth,                  # Encoder depth number integer
    activation = 'name_of_my_function'      # Sigmoïd, Softmax, Logsoftmax, Tanh, Identity, ...
)
```

We can then add data augmentation if we want to fool the neural network by applying several transformations to the images and masks so that the network is more reliable and able to generalize unseen data. We then define the learning parameters using a semantic segmentation model based on the previously created network.
```
segmentation_model = SegmentationModel(
    model = my_model,                       # Put the name of your model
    optimizer_name = 'my_optimizer',        # Adam, AdamW, RMSprop, SGD, ...
    log_every_n_steps = n,                  # Control Logging Frequency
    learning_rate = learning_rate,          # Learning Rate Scheduling
    loss = 'my_loss',                       # CombinedLoss, DiceLoss, JaccardLoss, TverskyLoss, ...
    data_augmentation = data_augmentation   # Dictionary containing all the transformations to be applied during training
)
```

We can choose whether or not to use weights already trained for building detection using spatial imagery, by entering the direction to the .ckpt file.
```
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',                                             # Quantity to monitor : 'val_loss', 'loss', ...
    filename='my_best_model',                                       # Save parameter in this directory
    save_top_k=1,                                                   # the best k models according to the quantity monitored will be saved
    mode='min'                                                      # If save_top_k != 0, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity
)
name_best_model = 'Pre_train_weights.ckpt'                          
segmentation_model.load_state_dict(torch.load(name_best_model))     # Load pre-train weights with model
```

## 2. Inference

