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
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, recall_score, precision_score, f1_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import keras.backend as K

class ModelEvaluator:
    def __init__(self, model, x_test, y_test, code):
        """
        Initializes the ModelEvaluator with the model, test data, and the label encoding.

        :param model: The trained model for predictions
        :param x_test: The test features
        :param y_test: The true labels for the test data
        :param code: A dictionary that maps class labels to numeric values
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.code = code
        self.labels = list(code.values())
        self.target_names = list(code.keys())
        
    def evaluate(self):
        """
        Evaluates the model and prints out the classification report,
        confusion matrix, and precision, recall, f1-score.
        """
        # Predicting the class probabilities
        y_pred = self.model.predict(self.x_test)
        
        # Getting the predicted class labels
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Print classification report
        self.print_classification_report(y_pred_classes)

        # Print precision, recall, and f1-score
        self.print_precision_recall_fscore(y_pred_classes)

        # Generate and plot confusion matrix and metrics
        self.generate_confusion_matrix_and_metrics(y_pred_classes)

    def print_classification_report(self, y_pred_classes):
        """
        Prints the classification report, which includes precision, recall, f1-score for each class.

        :param y_pred_classes: Predicted class labels
        """
        report = classification_report(self.y_test, y_pred_classes, labels=self.labels, target_names=self.target_names)
        print("Classification Report:")
        print(report)

    def print_precision_recall_fscore(self, y_pred_classes):
        """
        Prints the precision, recall, and f1-score for the model using a weighted average.

        :param y_pred_classes: Predicted class labels
        """
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            self.y_test, y_pred_classes, labels=self.labels, average='weighted'
        )
        
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1_score:.4f}')
        
    def generate_confusion_matrix_and_metrics(self, y_pred_classes):
        """
        Generate confusion matrix and classification metrics, then plot the confusion matrix.

        :param y_pred_classes: Predicted class labels
        """
        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)

        # Plot confusion matrix
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.target_names, 
                    yticklabels=self.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

        # Classification report
        report = classification_report(self.y_test, y_pred_classes, labels=self.labels, target_names=self.target_names)
        print("Classification Report:")
        print(report)

        # Precision, Recall, F1-score
        precision, recall, f1_score, _ = precision_recall_fscore_support(self.y_test, y_pred_classes, labels=self.labels, average='weighted')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1_score:.4f}')

        return cm, report, precision, recall, f1_score
        
class ROCCurvePlotter:
    def __init__(self, model, x_test, y_test, code):
        """
        Initializes the ROCCurvePlotter with the model, test data, and label encoding.

        :param model: The trained model for predictions
        :param x_test: The test features
        :param y_test: The true labels for the test data
        :param code: A dictionary that maps class labels to numeric values
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.code = code
        self.labels = list(code.values())
        self.target_names = list(code.keys())
    
    def plot_roc_curve(self):
        """
        Plots the ROC curve for each class and computes the AUC score.
        Handles both binary and multi-class classification.
        """
        # Check if it's binary classification (2 unique classes)
        if len(self.labels) == 2:
            self._plot_binary_roc_curve()
        else:
            self._plot_multiclass_roc_curve()

    def _plot_binary_roc_curve(self):
        """
            Plots the ROC curve for binary classification and computes the AUC score.
        """

        # Ensure consistent sample sizes
        assert len(self.y_test) == len(self.x_test), "y_test and x_test must have the same length"

        # Binarize the true labels
        y_test_bin = label_binarize(self.y_test, classes=self.labels)

        # Predict probabilities using `predict_proba()` for binary classification
        y_pred = self.model.predict(self.x_test)[:, 1]  # Extract probabilities for the positive class

        # Extract class probabilities (assuming binary classification)
        y_pred_prob = y_pred.ravel()  # Assuming predictions are for positive class (index 1)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test_bin, y_pred_prob)

        # Compute AUC score
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'Binary Class AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Binary Classification)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def _plot_multiclass_roc_curve(self):
        """
        Plots the ROC curve for multi-class classification and computes the AUC score for each class.
        """
        # Binarize the true labels for multi-class ROC
        y_test_bin = label_binarize(self.y_test, classes=self.labels)
        
        # Predict class probabilities for all classes
        y_pred_prob = self.model.predict(self.x_test)
        
        # Initialize plot
        plt.figure(figsize=(10, 8))
        
        # Loop through each class to calculate and plot the ROC curve
        for i in range(len(self.labels)):
            # Compute ROC curve for each class
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            # Compute AUC score
            roc_auc = auc(fpr, tpr)
            
            # Plot the ROC curve for the current class
            plt.plot(fpr, tpr, lw=2, label=f'{self.target_names[i]} (AUC = {roc_auc:.2f})')
        
        # Plot the diagonal line (chance line)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Add labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multi-Class Classification')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Show the plot
        plt.show()
