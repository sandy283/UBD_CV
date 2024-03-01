import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from skimage import io, color, exposure, segmentation
import matplotlib.pyplot as plt
from glob import glob

# Define the paths to the image and mask folders
image_folder = r'F:\UBD_CV\UNET\images'
mask_folder = r'F:\UBD_CV\UNET\masks'

# Function to load and preprocess images and masks
def load_data(image_folder, mask_folder, num_images):
    images = []
    masks = []
    seg_masks = []

    # List and sort the image files
    image_files = sorted(glob(os.path.join(image_folder, '*.jpg')))
    mask_files = sorted(glob(os.path.join(mask_folder, '*.jpg')))

    for i in range(num_images):
        image_path = image_files[i]
        mask_path = mask_files[i]

        # Read the image for mean shift segmentation
        img = io.imread(image_path)
        img_lab = color.rgb2lab(img)
        img_lab = exposure.rescale_intensity(img_lab, in_range=(0, 1))
        seg_mask = segmentation.felzenszwalb(img_lab, scale=6, sigma=4.5, min_size=1000)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Preprocess images and masks as needed
        image = image / 255.0  # Normalize images to the range [0, 1]
        mask = mask / 255.0  # Normalize masks to the range [0, 1]

        images.append(image)
        masks.append(mask)
        seg_masks.append(seg_mask)

    return np.array(images), np.array(masks), np.array(seg_masks)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_flat = keras.backend.flatten(y_true)
    y_pred_flat = keras.backend.flatten(y_pred)
    
    intersection = keras.backend.sum(y_true_flat * y_pred_flat)
    union = keras.backend.sum(y_true_flat) + keras.backend.sum(y_pred_flat)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Load training data
num_training_images = 100
images, masks, seg_masks = load_data(image_folder, mask_folder, num_training_images)

# Add mean shift segmentation mask as an additional channel
images_with_seg = np.concatenate([images, seg_masks[..., None]], axis=-1)



seg_output_folder = r'F:\UBD_CV\UNET\segmented_images_mean_shift'
os.makedirs(seg_output_folder, exist_ok=True)

for i in range(num_training_images):
    segmented_image = seg_masks[i]
    
    seg_output_path = os.path.join(seg_output_folder, f'segmented_image_{i + 1}.jpg')
    plt.imsave(seg_output_path, segmented_image, cmap='gray')

    print(f"Segmented image from mean shift saved: {seg_output_path}")




model = keras.Sequential([
    layers.Input(shape=(None, None, 4)),  # Allow variable image size with segmentation mask
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    
    # ... Add more layers as needed
    
    # Use transposed convolutions to upsample the spatial resolution
    layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')  # Modify this line
])

model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', dice_coef])

model.fit(images_with_seg, masks, epochs=50, batch_size=4, validation_split=0.1)

model.save('increased_model_with_mean_shift.h5')
