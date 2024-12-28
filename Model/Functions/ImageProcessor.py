import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import cv2
import random
import glob as gb
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet152V2, VGG16, VGG19, InceptionV3, EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, recall_score, precision_score, f1_score
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm
import keras.backend as K


class ImageReader:
    def __init__(self, base_path):
        """
        Initializes the ImageReader class with the base path where images are located.
        
        Parameters:
        - base_path: The path containing images or subfolders with images.
        """
        self.base_path = base_path

    def count_images_in_folders(self, description, is_subfolder=False):
        """
        Counts and prints the number of images in each folder within the specified base path.

        Parameters:
        - description: A string describing the type of data (e.g., 'training', 'testing', or 'prediction').
        - is_subfolder: Set to True if you want to count images in all subfolders.
        """
        if is_subfolder:
            # If we want to search through subfolders
            for folder in os.listdir(self.base_path):
                folder_path = os.path.join(self.base_path, folder)
                if os.path.isdir(folder_path):  # Ensure it's a directory
                    images = gb.glob(os.path.join(folder_path, '*.jpg'))
                    print(f'For {description} data, found {len(images)} images in folder "{folder}".')
        else:
            # Only count images in the immediate base folder
            files = gb.glob(os.path.join(self.base_path, '*.jpg'))
            print(f'For {description} data, found {len(files)} images in the base folder.')

    def get_image_sizes(self, is_subfolder=False):
        """
        This function calculates the sizes of images in the given path.
        
        Parameters:
        - is_subfolder: Set to True if the images are in subfolders (like train/test sets).
        
        Returns:
        - A pandas Series with the count of each image size.
        """
        sizes = []

        if is_subfolder:
            for folder in os.listdir(self.base_path):
                folder_path = os.path.join(self.base_path, folder)
                if os.path.isdir(folder_path):  # Ensure it's a directory
                    images = gb.glob(pathname=os.path.join(folder_path, '*.jpg'))
                    for img in images:
                        try:
                            image = plt.imread(img)
                            sizes.append(image.shape)
                        except Exception as e:
                            print(f"Could not read {img}: {e}")
        else:
            files = gb.glob(pathname=os.path.join(self.base_path, '*.jpg'))
            for file in files:
                try:
                    image = plt.imread(file)
                    sizes.append(image.shape)
                except Exception as e:
                    print(f"Could not read {file}: {e}")

        return pd.Series(sizes).value_counts()
    
class ImageAugmentation:
    def __init__(self, train_path, augmented_train_path, target_count, image_size):
        self.train_path = train_path
        self.augmented_train_path = augmented_train_path
        self.target_count = target_count
        self.image_size = image_size
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def resize_and_augment_images(self, class_name):
        class_folder = os.path.join(self.train_path, class_name)
        augmented_class_folder = os.path.join(self.augmented_train_path, class_name)

        if not os.path.exists(augmented_class_folder):
            os.makedirs(augmented_class_folder)

        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        current_image_count = len(image_files)

        if current_image_count >= self.target_count:
            print(f"Skipping {class_name}: Already has {current_image_count} images.")
            return

        needed_augmentations = self.target_count
        print(f"Class {class_name} has {current_image_count} images. {needed_augmentations} needed.")
        max_augmentations_per_image = int(np.ceil(needed_augmentations / current_image_count))
        min_augmentations_per_image = int(np.floor(needed_augmentations / current_image_count))

        aug_count = 0
        
        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            image = load_img(image_path, target_size=self.image_size)
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            augmentations_per_image = random.randint(min_augmentations_per_image , max_augmentations_per_image + 1)

            for j in range(augmentations_per_image):
                # Generate a batch of augmented images
                augmented_images = self.datagen.flow(image_array, batch_size=1)

                # Get the first image from the batch
                augmented_image = next(augmented_images)[0]

                # Convert augmented image from RGB to BGR (for OpenCV saving)
                augmented_image = cv2.cvtColor(augmented_image.astype('uint8'), cv2.COLOR_RGB2BGR)

                # Save the augmented image
                augmented_image_path = os.path.join(augmented_class_folder, f"{os.path.splitext(image_file)[0]}_aug_{aug_count}.jpg")
                cv2.imwrite(augmented_image_path, augmented_image)
                aug_count += 1

                if aug_count >= needed_augmentations:
                    print(f"Augmentation completed for {class_name}: {aug_count} images added.")
                    break

            if aug_count >= needed_augmentations:
                break

        print(f"Augmentation completed for {class_name}: {aug_count} images added.")

    def augment_all_classes(self):
        class_names = [f for f in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, f))]

        for class_name in class_names:
            print(f"Processing class: {class_name}")
            self.resize_and_augment_images(class_name)

class ImageLoading:
    def __init__(self):
        """
        Initializes the ImageLoading class for loading images with resizing and label encoding.
        """
        self.label_encoder = LabelEncoder()  # To encode folder names to numeric labels
    
    def shuffle_data(self, x, y):
        # Combine x and y, shuffle, then split back
        combined = list(zip(x, y))
        np.random.shuffle(combined)  # Shuffle the combined list
        x_shuffled, y_shuffled = zip(*combined)  # Unzip the combined list into x and y
        return np.array(x_shuffled), np.array(y_shuffled)

    def load_images(self, image_path, code=None, is_subfolder=True):
        """
        Loads images from the specified path, resizes them to 224x224, and encodes labels.
        
        Parameters:
        - image_path: Path where images are stored.
        - code: A dictionary mapping folder names to labels (only required for training/test data).
        - is_subfolder: Boolean indicating if images are in subfolders (True for train/test, False for prediction).
        
        Returns:
        - x_data: A NumPy array of resized images, normalized to [0, 1].
        - y_data: Corresponding labels (if applicable, else returns None for prediction data).
        """
        x_data = []
        y_data = []

        if is_subfolder:
            if code is None:
                raise ValueError("Code dictionary must be provided for train/test data loading.")
            
            # Encode the labels
            folder_names = list(code.keys())
            self.label_encoder.fit(folder_names)
            
            for folder in os.listdir(image_path):
                folder_path = os.path.join(image_path, folder)
                if os.path.isdir(folder_path):  # Ensure the path is a folder
                    images = gb.glob(pathname=os.path.join(folder_path, '*.jpg'))
                    for img_path in images:
                        # Open the image using PIL
                        image = Image.open(img_path)
                        image = image.resize((224, 224))  # Resize the image to 224x224
                        image = np.array(image)  # Convert to numpy array
                        x_data.append(image)
                        y_data.append(self.label_encoder.transform([folder])[0])  # Encode the label
        else:
            # For prediction data (without labels)
            images = gb.glob(pathname=os.path.join(image_path, '*.jpg'))
            for img_path in images:
                # Open the image using PIL
                image = Image.open(img_path)
                image = image.resize((224, 224))  # Resize the image to 224x224
                image = np.array(image)  # Convert to numpy array
                x_data.append(image)
            y_data = None  # No labels for prediction data

        # Convert x_data and y_data to numpy arrays
        x_data = np.array(x_data, dtype='float32') / 255  # Normalize the images
        if y_data is not None:
            y_data = np.array(y_data, dtype='int')  # Convert labels to integers

        return x_data, y_data