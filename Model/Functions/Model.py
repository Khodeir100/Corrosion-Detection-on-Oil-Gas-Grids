import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import cv2
import glob as gb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet152V2, VGG16, VGG19, InceptionV3, EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, recall_score, precision_score, f1_score
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import keras.backend as K

# Define your model architectures
class ModelArchitectures:
    def __init__(self, input_shape=(224, 224, 3), weights='imagenet'):
        """
        Initializes the ModelArchitectures class to build different models.
        
        Parameters:
        - input_shape: Shape of the input images (default is (224, 224, 3)).
        - weights: Pre-trained weights to load (default is 'imagenet').
        """
        self.input_shape = input_shape
        self.weights = weights

    def vgg16(self):
        """
        Returns the VGG16 model with ImageNet weights, excluding the top layers.
        """
        model = tf.keras.applications.VGG16(weights=self.weights, include_top=False, input_shape=self.input_shape)
        model.trainable = False  # Freeze layers to start
        return model

    def vgg19(self):
        """
        Returns the VGG19 model with ImageNet weights, excluding the top layers.
        """
        model = tf.keras.applications.VGG19(weights=self.weights, include_top=False, input_shape=self.input_shape)
        model.trainable = False  # Freeze layers to start
        return model

    def resnet152v2(self):
        """
        Returns the ResNet152V2 model with ImageNet weights, excluding the top layers.
        """
        model = tf.keras.applications.ResNet152V2(weights=self.weights, include_top=False, input_shape=self.input_shape)
        model.trainable = False  # Freeze layers to start
        return model

    def inception_v3(self):
        """
        Returns the InceptionV3 model with ImageNet weights, excluding the top layers.
        """
        model = tf.keras.applications.InceptionV3(weights=self.weights, include_top=False, input_shape=self.input_shape)
        model.trainable = False  # Freeze layers to start
        return model

    def efficientnet(self):
        """
        Returns the EfficientNetB0 model with ImageNet weights, excluding the top layers.
        """
        model = tf.keras.applications.EfficientNetB0(weights=self.weights, include_top=False, input_shape=self.input_shape)
        model.trainable = False  # Freeze layers to start
        return model

# Define ModelTrainer class
class ModelTrainer:
    def __init__(self, models, class_labels, train_data, val_data, test_data, epochs=10, batch_size=32, checkpoint_path="model_checkpoint", validation_batch_size=5):
        """
        Initializes the ModelTrainer class to train and evaluate different models.
        
        Parameters:
        - models: A dictionary of model functions.
        - class_labels: List of class labels for classification.
        - train_data: Tuple (x_train, y_train) for training data.
        - val_data: Tuple (x_val, y_val) for validation data.
        - test_data: Tuple (x_test, y_test) for testing data.
        - epochs: Number of epochs to train.
        - batch_size: Batch size for training.
        - checkpoint_path: Path to save the best model during training.
        """
        self.models = models
        self.class_labels = class_labels
        self.x_train, self.y_train = train_data
        self.x_val, self.y_val = val_data
        self.x_test, self.y_test = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.checkpoint_path = checkpoint_path
        
        # Define callbacks
        self.early = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, mode='max')
        self.lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=2, min_lr=1e-6, verbose=1,mode='max')
        self.checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
        self.lr_scheduler = LearningRateScheduler(self.lr_schedule)

        self.current_lr = 0.0004


    def lr_schedule(self, epoch):

        min_lr = 1e-6
        drop_factor = 0.8
        drop_every_n_epochs = 4

        if epoch % drop_every_n_epochs == 0 and epoch != 0:
            self.current_lr *= drop_factor
        return max(self.current_lr, min_lr)

    def build_model(self, model_func):
        """
        Builds a model by adding a classification head to the base model.
        
        Parameters:
        - model_func: Function that returns the base model architecture.
        
        Returns:
        - model: The complete model with the classification head.
        """
        base_model = model_func()  # Get the base model (e.g., VGG16)
        
        # Add the classification head to the model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.35),
            BatchNormalization(),
            Dense(384, activation='relu',kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.25),
            Dense(192, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(len(self.class_labels), activation='softmax')  # Number of output classes
        ])
        
        return model

    def compile_and_train(self, model):
        """
        Compiles and trains the model.
        
        Parameters:
        - model: The complete model to train.
        
        Returns:
        - history: The history of training.
        """
        model.compile(optimizer=Adam(clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model using the filtered data
        history = model.fit(
            self.x_train,
            tf.keras.utils.to_categorical(self.y_train, num_classes=len(self.class_labels)),
            validation_data=(self.x_val, tf.keras.utils.to_categorical(self.y_val, num_classes=len(self.class_labels))),
            validation_batch_size=5,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.early, self.lr_reducer, self.checkpoint, self.lr_scheduler]
        )
        
        return history

    def evaluate_model(self, model):
        """
        Evaluates the model on the test set.
        
        Parameters:
        - model: The trained model to evaluate.
        
        Returns:
        - loss: The loss value on the test set.
        - accuracy: The accuracy on the test set.
        """
        loss, accuracy = model.evaluate(self.x_test, tf.keras.utils.to_categorical(self.y_test, num_classes=len(self.class_labels)))
        return loss, accuracy

    def plot_all_models_metrics(self, histories):
        """
            Plots the loss and accuracy curves for all models on the same figure with two subplots.
    
            Parameters:
             - histories: Dictionary of model names as keys and their corresponding training histories as values.
        """
        # Create a figure with 1 row and 2 columns for subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        save_path = 'D:/Corrosion detection/Corrosion Detection/fig'
    
         # Plotting loss for all models on the left subplot (ax1)
        for model_name, history in histories.items():
            # Plotting training loss
            ax1.plot(history.history['loss'], label=f'{model_name} Train Loss')
            # Plotting validation loss
            ax1.plot(history.history['val_loss'], label=f'{model_name} Val Loss')
    
         # Plotting accuracy for all models on the right subplot (ax2)
        for model_name, history in histories.items():
            # Plotting training accuracy
             ax2.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
              # Plotting validation accuracy
             ax2.plot(history.history['val_loss'], label=f'{model_name} Val Accuracy')
    
        # Set labels, titles, and legends for each subplot
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend(loc='upper right')

        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend(loc='upper right')
    
        # Adjust the layout to prevent overlapping
        plt.tight_layout()

        # Save the figure
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

        # Show the plot
        plt.show()

    def train_and_evaluate_all_models(self):
        """
        Trains and evaluates all models in the models dictionary.
        """
        histories = {}
        for model_name, model_func in self.models.items():
            print(f"Training {model_name}...")

            # Build and compile the model
            model = self.build_model(model_func)
            
            # Train the model
            history = self.compile_and_train(model)
            
            print(f"Evaluating {model_name}...")
            loss, accuracy = self.evaluate_model(model)
            print(f"{model_name} evaluation complete.")
            print(f"Loss: {loss}, Accuracy: {accuracy}")
            print("_______________________________________________________________________")
            
            # Store the training history for plotting later
            histories[model_name] = history

        # After training all models, plot their metrics on the same graph
        self.plot_all_models_metrics(histories)