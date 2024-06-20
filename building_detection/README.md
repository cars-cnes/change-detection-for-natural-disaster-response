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

We can then add data augmentation if we want to fool the neural network by applying several transformations to the images and masks so that the network is more reliable and able to generalize unseen data. We then define the learning parameters using a semantic segmentation model based on the previously created network :
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
    monitor = 'val_loss',                                             # Quantity to monitor : 'val_loss', 'loss', ...
    filename = 'my_best_model',                                       # Save parameter in this directory
    save_top_k = 1,                                                   # the best k models according to the quantity monitored will be saved
    mode = 'min'                                                      # If save_top_k != 0, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity
)
name_best_model = 'Pre_train_weights.ckpt'                          
segmentation_model.load_state_dict(torch.load(name_best_model))     # Load pre-train weights with model
```

Before you can run the training with the dataset, you need to set the trainer conditions :
```
trainer = pl.Trainer(
    callbacks = [checkpoint_callback],  # Save checkpoints
    max_epochs = number_epochs,         # Maximum number of epochs for training and validation stage
    log_every_n_steps = n,              # Control Logging Frequency
    logger = logger,                    # Training and validation stage control with tensorboard
    devices = n_devices,                # Number of device (CPU, GPU, ...)
    accelerator = "my_accelerator"      # CPU, GPU, ...
)
```

Once all these steps have been carried out and parameterized, we can start training our neural network on the dataset and then save the best training weights :
```
trainer.fit(segmentation_model, datamodule=data_module)             # Training and validation stages for semantic segmentation model on dataset

torch.save(segmentation_model.state_dict(), 'best_model.ckpt')      # Save best weights for this model configuration
```

### Training for FOSS4G

In order to be able to make inferences on CNES Pleiades data, we propose to choose the best training parameters and weights on xView2 data that favor building recognition on Pleiades data in the validation stage :
```
model = get_model(
    num_classes = 1, 
    neural_network = 'MAnet',
    encoder_name = 'efficientnet-b7',
    encoder_depth = 5,
    activation = 'sigmoid'
)

data_augmentation = {
    # Geometric transformations
    'RandomRotation': (0, 180),
    'RandomVerticalFlip': 0.5,
    'RandomHorizontalFlip': 0.5,
    'RandomCrop': (256, 256),

    # Radiometric transformations
    'ColorJitter': {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1 
    },
    'RandomGamma': {
        'gamma': (0.7, 1.5),
        'p': 0.5
    },
    'RandomBrightness': {
        'brightness': (0.5, 1.5),
        'p': 0.5
    },
    'RandomContrast': {
        'contrast': (0.5, 1.5),
        'p': 0.5
    },
    'GaussianBlur': {
        'kernel_size': (5, 5),
        'sigma': (0.1, 2.0)
    }
}

segmentation_model = SegmentationModel(
    model = model,
    optimizer_name = 'AdamW',
    log_every_n_steps = 30,
    learning_rate = 0.0001,
    loss = 'CombinedLoss',
    data_augmentation = data_augmentation
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = 'val_loss',
    filename = 'best_model',
    save_top_k = 1,
    mode = 'min'
)

trainer = pl.Trainer(
    callbacks = [checkpoint_callback], 
    max_epochs = 50, 
    log_every_n_steps = 30,
    logger = logger,
    devices = 1, 
    accelerator = "auto"
)
```

## 2. Inference

The aim of the inference is to be able to apply the neural network trained in building detection from satellite images to CNES Pleiades images. We therefore use the weights from the previous training to apply them to the images in the chart on the earthquake in Turkey (seen on the tutorial.ipynb). So, first, an image of any size is fed into the neural network. The network will predict a mask highlighting the potential buildings it has recognized in the image. This prediction mask is then superimposed on the input image, showing all the buildings that the network has detected.