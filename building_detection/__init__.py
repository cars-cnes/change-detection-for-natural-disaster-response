import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import segmentation_models_pytorch as smp
import kornia.augmentation as K
import torchvision.transforms as T
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
import matplotlib.pyplot as plt

def pad_image(image):
    H, W, C = image.shape
    
    H_target = ((H + 31) // 32) * 32
    W_target = ((W + 31) // 32) * 32

    pad_h = H_target - H
    pad_w = W_target - W
    
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    
    return padded_image

def remove_padding(padded_image, original_size):
    H_original, W_original = original_size
    
    unpadded_image = padded_image[:H_original, :W_original]
    
    return unpadded_image

def normalize_image_percentiles(image, lower_percentile=2, upper_percentile=98):
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    
    image_clipped = np.clip(image, lower_bound, upper_bound)
    
    image_normalized = (image_clipped - lower_bound) / (upper_bound - lower_bound)
    
    image_scaled = (image_normalized * 255).astype(np.uint8)
    
    return image_normalized

def load_data(image_dir) :
    images = []
    with rasterio.open(image_dir) as img_file:
        image = img_file.read() 
        image = reshape_as_image(image) 
        image = normalize_image_percentiles(image)
        images.append(image[:,:,:3])

    return np.array(images)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return dice + bce

def iou_score(y_pred, y_true, smooth=1):
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, data_augmentation) :
        super(DataAugmentation, self).__init__()

        # Initialize geometric transformations list
        geometric_transformations = []
        if 'RandomRotation' in data_augmentation:
            geometric_transformations.append(K.RandomRotation(degrees=data_augmentation['RandomRotation']))
        if 'RandomVerticalFlip' in data_augmentation:
            geometric_transformations.append(K.RandomVerticalFlip(p=data_augmentation['RandomVerticalFlip']))
        if 'RandomHorizontalFlip' in data_augmentation:
            geometric_transformations.append(K.RandomHorizontalFlip(p=data_augmentation['RandomHorizontalFlip']))
        if 'RandomCrop' in data_augmentation:
            self.crop_size = data_augmentation['RandomCrop']
            geometric_transformations.append(K.RandomCrop(size=self.crop_size))
        if 'RandomAffine' in data_augmentation:
            geometric_transformations.append(K.RandomAffine(
                degrees=data_augmentation['RandomAffine']['degrees'],
                translate=data_augmentation['RandomAffine']['translate'],
                scale=data_augmentation['RandomAffine']['scale'],
                shear=data_augmentation['RandomAffine']['shear']
            ))
        if 'RandomPerspective' in data_augmentation:
            geometric_transformations.append(K.RandomPerspective(
                distortion_scale=data_augmentation['RandomPerspective']['distortion_scale'],
                p=data_augmentation['RandomPerspective']['p']
            ))

        self.geometric_transform = nn.Sequential(*geometric_transformations)

        # Initialize radiometric transformations list
        radiometric_transformations = []
        if 'ColorJitter' in data_augmentation:
            radiometric_transformations.append(K.ColorJitter(
                brightness=data_augmentation['ColorJitter']['brightness'],
                contrast=data_augmentation['ColorJitter']['contrast'],
                saturation=data_augmentation['ColorJitter']['saturation'],
                hue=data_augmentation['ColorJitter']['hue']
            ))
        if 'RandomGamma' in data_augmentation:
            radiometric_transformations.append(K.RandomGamma(
                gamma=data_augmentation['RandomGamma']['gamma'],
                p=data_augmentation['RandomGamma']['p']
            ))
        if 'RandomGrayscale' in data_augmentation:
            radiometric_transformations.append(K.RandomGrayscale(p=data_augmentation['RandomGrayscale']))
        if 'RandomBrightness' in data_augmentation:
            radiometric_transformations.append(K.RandomBrightness(
                brightness=data_augmentation['RandomBrightness']['brightness'],
                p=data_augmentation['RandomBrightness']['p']
            ))
        if 'RandomContrast' in data_augmentation:
            radiometric_transformations.append(K.RandomContrast(
                contrast=data_augmentation['RandomContrast']['contrast'],
                p=data_augmentation['RandomContrast']['p']
            ))
        if 'RandomSharpness' in data_augmentation:
            radiometric_transformations.append(K.RandomSharpness(
                sharpness=data_augmentation['RandomSharpness']['sharpness'],
                p=data_augmentation['RandomSharpness']['p']
            ))
        if 'GaussianBlur' in data_augmentation:
            radiometric_transformations.append(T.GaussianBlur(
                kernel_size=data_augmentation['GaussianBlur']['kernel_size'],
                sigma=data_augmentation['GaussianBlur']['sigma']
            ))

        self.radiometric_transform = nn.Sequential(*radiometric_transformations)

    def forward(self, images: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        seed = torch.randint(0, 1000000, (1,))
        torch.manual_seed(seed)

        # Concatenate images and masks along the channel dimension
        inputs = torch.cat((images, masks), dim=1)

        # Apply the same random transformation to both images and masks
        transformed_inputs = self.transform(inputs, seed.item())

        # Split the transformed inputs back into images and masks
        transformed_images, transformed_masks = torch.split(transformed_inputs, images.shape[1], dim=1)

        # Apply radiometric transformations only on images
        transformed_images = self.radiometric_transform(transformed_images)

        return transformed_images, transformed_masks

    def transform(self, inputs: Tensor, seed: int) -> Tensor:
        # Apply the random geometric transformations
        random_transformed = self.geometric_transform(inputs)
        return random_transformed

class CustomDataset(Dataset):
    def __init__(self, images, masks, data_augmentation):
        self.images = images                                 
        self.masks = masks
        self.target_size = data_augmentation['RandomCrop']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        threshold = 0
        mask = (mask > threshold).astype(np.float32)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        # Apply random cropping
        image_crop, mask_crop = self.random_crop(image_tensor, mask_tensor)

        return image_crop, mask_crop

    def random_crop(self, image, mask):
        h, w = self.target_size
        th, tw = image.size(1), image.size(2)
        if w == tw and h == th:
            return image, mask
        i = torch.randint(0, th - h + 1, (1,))
        j = torch.randint(0, tw - w + 1, (1,))
        image_crop = TF.crop(image, i.item(), j.item(), h, w)
        mask_crop = TF.crop(mask, i.item(), j.item(), h, w)
        return image_crop, mask_crop

class SegmentationModel(pl.LightningModule):
    def __init__(self, model=None, optimizer_name=None, learning_rate=None, log_every_n_steps=None, loss=None, data_augmentation=None, **optimizer_params):
        super().__init__()
        self.optimizer_name = optimizer_name 
        self.learning_rate = learning_rate
        self.optimizer_params = optimizer_params
        self.log_every_n_steps = log_every_n_steps
        self.automatic_optimization = True
        self.data_augmentation = data_augmentation

        self.training_step_outputs_loss = []
        self.training_step_outputs_iou = []

        self.transform = DataAugmentation(data_augmentation)
        self.model = model.to('cpu')

        if loss == 'DiceLoss' :
            self.loss = DiceLoss()
        elif loss == 'bce_loss' :
            self.loss = nn.BCEWithLogitsLoss() 
        elif loss == 'CombinedLoss' :
            self.loss= CombinedLoss()

        self.log_dir = log_dir
        self.tb_writer = logger

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

        # Change the learning rate
        scheduler = CosineAnnealingLR(optimizer, T_max=20)

        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training:
            x, y = self.transform(x, y)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        iou = iou_score(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)

        self.training_step_outputs_loss.append(loss)
        self.training_step_outputs_iou.append(iou)

        self.log_images(x, y, y_hat, 'train', self.global_step)

        return {"loss": loss, "iou": iou}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        val_loss = self.loss(y_hat, y)
        iou = iou_score(y_hat, y)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

        self.log_images(x, y, y_hat, 'val', self.global_step)

        return {"val_loss": val_loss, "val_iou": iou}

    def log_images(self, x, y, y_hat, tag, step):
        # Image grid creation
        input_grid = make_grid(x[:3], nrow=3, normalize=True)
        target_grid = make_grid(y[:3], nrow=3, normalize=True)
        predicted_grid = make_grid(y_hat[:3], nrow=3, normalize=True)

        # Log images using SummaryWriter's
        self.tb_writer.experiment.add_image(f'{tag}/input_images', input_grid, global_step=step)
        self.tb_writer.experiment.add_image(f'{tag}/target_masks', target_grid, global_step=step)
        self.tb_writer.experiment.add_image(f'{tag}/predicted_masks', predicted_grid, global_step=step)

    def on_train_epoch_end(self):
        epoch_mean_loss = torch.stack(self.training_step_outputs_loss).mean()
        self.log("loss", epoch_mean_loss, prog_bar=True)

        epoch_mean_iou = torch.stack(self.training_step_outputs_iou).mean()
        self.log("iou", epoch_mean_iou, prog_bar=True)

        self.training_step_outputs_loss.clear()
        self.training_step_outputs_iou.clear()

class DataModule(pl.LightningDataModule):
    def __init__(self, images, masks, data_augmentation, batch_size=32):
        super().__init__()
        self.images = images
        self.masks = masks
        self.data_augmentation = data_augmentation
        self.batch_size = batch_size

    def setup(self, stage=None):
        total_size = len(self.images)
        train_size = int(0.87 * total_size)
        val_size = total_size - train_size

        self.train_dataset = CustomDataset(self.images[:train_size], self.masks[:train_size], self.data_augmentation)
        self.val_dataset = CustomDataset(self.images[train_size:], self.masks[train_size:], self.data_augmentation) 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=10)

def get_model(num_classes, neural_network, encoder_name, encoder_depth, activation):
    if neural_network == 'Unet' :
        model = smp.Unet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'UnetPlusPlus' :
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'DeepLabV3' :
        model = smp.DeepLabV3(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'DeepLabV3Plus' :
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'MAnet' :
        model = smp.MAnet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'Linknet' :
        model = smp.Linknet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'FPN' :
        model = smp.FPN(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'PSPNet' :
        model = smp.PSPNet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'PAN' :
        model = smp.PAN(encoder_name=encoder_name, in_channels=3, classes=1, activation=activation)

    return model

log_dir = "/tensorboard/logs"
logger = pl.loggers.TensorBoardLogger(log_dir, name="change_detection")

data_augmentation = {
    # Geometric transformations
    'RandomRotation': (0, 180),
    'RandomVerticalFlip': 0.5,
    'RandomHorizontalFlip': 0.5,
    'RandomCrop': (256, 256),
    # 'RandomAffine': {
    #     'degrees': 10,
    #     'translate': (0.1, 0.1),
    #     'scale': (0.8, 1.2),
    #     'shear': 10
    # },
    # 'RandomPerspective': {
    #     'distortion_scale': 0.5,
    #     'p': 0.5
    # },

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
    # 'RandomGrayscale': 0.2,
    'RandomBrightness': {
        'brightness': (0.5, 1.5),
        'p': 0.5
    },
    'RandomContrast': {
        'contrast': (0.5, 1.5),
        'p': 0.5
    },
    # 'RandomSharpness': {
    #     'sharpness': (0.5, 2.0),
    #     'p': 0.5
    # },
    'GaussianBlur': {
        'kernel_size': (5, 5),
        'sigma': (0.1, 2.0)
    }
}

def overlay_mask(image, mask, color=(0.0, 1.0, 0.0), alpha=0.5):
    """
    Superimposes a mask on an image with a given color.
    """
    color = np.array(color)
    overlay = image.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    return overlay

def plot_prediction(initial_image, predicted_mask) :
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(initial_image[0])
    plt.title("Initial RGB image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted mask")
    plt.axis('off')

    buildings = predicted_mask > 0.5
    result = overlay_mask(initial_image[0], buildings, color=(0, 1, 0), alpha=0.5)

    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title("Image with Buildings detection")
    plt.axis('off')

    plt.show()

    return result

###############################################################################

def run(img_dir, weights):

    model = get_model(
    num_classes=1, 
    neural_network='MAnet',
    encoder_name='efficientnet-b7',
    encoder_depth=5,
    activation=None
    )

    segmentation_model = SegmentationModel(
        model=model,
        optimizer_name='AdamW',
        log_every_n_steps=30,
        learning_rate=0.0001,
        loss='CombinedLoss',
        data_augmentation=data_augmentation
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filename='best_model',
        save_top_k=1,
        mode='min'
    )

    image = load_data(img_dir)
    image_pad = np.array([pad_image(img) for img in image])

    checkpoint = torch.load(weights)
    segmentation_model.load_state_dict(checkpoint)
    segmentation_model.eval()  

    input_image = image_pad.transpose(0, 3, 1, 2)
    input_image = torch.from_numpy(input_image).float()

    with torch.no_grad():
        predicted_mask = segmentation_model(input_image)
        predicted_mask = predicted_mask.squeeze().cpu().numpy()

    predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

    predicted_mask_binary = remove_padding(predicted_mask_binary, image.shape[1:3])
    result = plot_prediction(image, predicted_mask_binary)
    
    return image[0], predicted_mask_binary, result

def training(images, masks, Pre_train_weights):
    data_module = DataModule(images, masks, data_augmentation)

    if Pre_train_weights != None :
        # Using a network with pre-trained weights in a .ckpt file
        name_best_model = 'Pre_train_weights/Pre_train_MAnet_efficientNet_b7_iou_0,7.ckpt'
        segmentation_model.load_state_dict(torch.load(name_best_model))

    # Drive configuration
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback], 
        max_epochs=10, 
        log_every_n_steps=10,
        logger=logger,
        devices=1, 
        accelerator="cpu"
    )

    # Model training with images and masks
    trainer.fit(segmentation_model, datamodule=data_module)

    # Save weights in a .ckpt file
    torch.save(segmentation_model.state_dict(), 'best_model.ckpt')

    print("training")
    return

if __name__ == "__main__":
    training()
