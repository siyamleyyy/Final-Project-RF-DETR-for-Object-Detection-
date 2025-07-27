# Final-Project-RF-DETR-for-Object-Detection-
RF-DETR for Hard Hat Detection
This repository contains a complete implementation of the RF-DETR (Receptive Field Enhanced Detection Transformer) model for object detection. The model is trained and evaluated on the public Hard Hat Detection dataset from Roboflow Universe to identify workers with and without proper head protection.

Project Overview
This project demonstrates an end-to-end object detection pipeline, from data acquisition and preprocessing to model training, evaluation, and visualization. The core of the project is the RF-DETR, a Transformer-based object detection model enhanced with a Receptive Field Enhancement (RFE) module. This custom module uses multi-scale convolutions and self-attention to improve the model's ability to recognize objects of varying sizes, a common challenge in detection tasks.

Key Features
Modern Architecture: Implements the DETR (Detection Transformer) architecture with a ResNet-50 backbone.

Novel Enhancement: Introduces a custom Receptive Field Enhancement (RFE) module to boost detection performance.

End-to-End Pipeline: Includes scripts for data loading, augmentation, training, and evaluation.

Bipartite Matching Loss: Utilizes a Hungarian Matcher to assign predictions to ground-truth boxes, a hallmark of DETR-based models.

Clear Evaluation: Calculates and displays standard object detection metrics, including mean Average Precision (mAP).

Rich Visualizations: Generates plots for loss curves and visualizes model predictions on test images.

Demo
(You can replace this image with a GIF of your model's predictions)

An example of the model detecting a 'helmet' and a 'head' on workers.

Installation
To get started with this project, clone the repository and set up the Python environment.

1. Clone the repository:

git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

2. Create and activate a virtual environment (recommended):

python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

3. Install the required dependencies:

pip install -r requirements.txt

(Note: A requirements.txt file can be generated with pip freeze > requirements.txt after installing the packages below if one is not already present.)

pip install roboflow torch torchvision opencv-python matplotlib scipy pillow pycocotools

Usage
Before running the script, you need to configure your Roboflow API key to download the dataset.

1. Get Your Roboflow API Key:

Log in to your Roboflow account.

Go to your workspace Settings and copy your Private API Key. This is crucial; the public key will not work for this script.

2. Fork the Dataset:

Go to the Hard Hat Detection project on Roboflow Universe.

Click "Fork Dataset" to copy it to your own workspace. This will give you the necessary permissions.

3. Update the Script:

Open the main Python script (Final_Exam.py or your script name).

Find the data loading section and replace the placeholders with your Private API Key, Workspace ID, and Project ID.

# Example from the script
rf = Roboflow(api_key="YOUR_PRIVATE_API_KEY")
project = rf.workspace("YOUR_WORKSPACE_ID").project("YOUR_PROJECT_ID")
version = project.version(3) # Or your forked version number
dataset = version.download("coco")

4. Run the Training Script:

Execute the main script from your terminal:

python Final_Exam.py

The script will automatically download the dataset, train the model, save the best weights to best_rf_detr_model.pth, and display evaluation results and visualizations.

Model Architecture
The RF-DETR model builds upon the standard DETR architecture with one key innovation:

CNN Backbone (ResNet-50): Extracts a rich feature map from the input image.

Receptive Field Enhancement (RFE) Module: This custom module takes the feature map and processes it with parallel 1x1, 3x3, and 5x5 convolutions. The results are fused and passed through a self-attention layer, allowing the model to dynamically adjust its receptive field to better capture features from both small and large objects.

Transformer Encoder-Decoder: The enhanced feature map is flattened and fed into the Transformer. The decoder then uses a set of learned object queries to attend to different parts of the image and output predictions.

Prediction Heads: Simple feed-forward networks at the end predict the class label and bounding box coordinates for each object query.

Dataset
This project uses the Hard Hat Detection dataset from Roboflow Universe. It contains images of construction workers and is annotated with three classes:

helmet: A person wearing a hard hat.

head: A person without a hard hat.

person: The bounding box for the person.

The dataset is automatically downloaded in COCO format by the script.
