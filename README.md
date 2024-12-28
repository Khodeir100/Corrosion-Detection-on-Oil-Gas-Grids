# Corrosion Detection For Oil & Gas Grids
 ## An application that uses advanced computer vision and deep learning techniques to detect and classify corrosion in oil and gas infrastructure.
 ### This corrosion detection system uses deep learning models trained on hundreds of images of gas infrastructure. The system:
 
##### 1. First identifies whether the component is a Pipeline or Flange.
##### 2. Then applies a specialized corrosion detection model based on the component type:
##### - For Pipelines: Detects Uniform Corrosion, No Corrosion, or Pitting
##### - For Flanges: Detects Crevice Corrosion or No Corrosion
##### 3. Provides confidence scores to help inform decision making.

## Here is a diagram illustrating the concept:
![Diagram](https://raw.githubusercontent.com/Khodeir100/Corrosion-Detection-on-Oil-Gas-Grids/main/App/Diagram.PNG)

# About Model:
## Dataset
####     - Original Dataset: 1,000 benchmark images, self-collected
####     - Augmented Dataset: 4,500 images (through data augmentation techniques)
####     - Hardware: Intel i7-6820HQ laptop with 16GB RAM
---
## Model Architecture Selection
####    - Initial Architectures Evaluated: (VGG16, VGG19, ResNet152V2, InceptionV3, EfficientNetB0)
####    - Best Performing Architecture: ResNet152V2 emerged as the most effective model for corrosion detection tasks.
---
## Training Configuration
#### Learning Rate: 0.0004 with ReduceLROnPlateau (adjusts learning rate when validation accuracy plateaus)
#### Batch Size: 32
#### Early Stopping: Patience of 4-6 epochs (to prevent overtraining)
#### Data Augmentation: Includes rotation, zoom, flip, and brightness adjustments to enhance dataset diversity and reduce overfitting.
#### Optimizer: Adam
#### L2 Regularization (0.001): Added to prevent overfitting and encourage better model generalization.
---
### Project Duration: 40 Days
---

#### The following image shows an example of uniform corrosion detected by the app deployed on Streamlit:

![Uniform Corrosion](https://raw.githubusercontent.com/Khodeir100/Corrosion-Detection-on-Oil-Gas-Grids/main/App/Uniform%20Corrosion.PNG)

