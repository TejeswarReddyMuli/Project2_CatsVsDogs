# Project2_CatsVsDogs_classification
🐶🐱 Cats vs Dogs Image Classification using CNN
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs.
The model learns visual patterns from images and predicts whether a given image belongs to a Cat or a Dog.

The project demonstrates an end-to-end deep learning workflow including data loading, preprocessing, augmentation, model training, evaluation, and visualization.

📂 Dataset

Dataset Name: Kaggle Cats vs Dogs Dataset

Source: Kaggle

Link:
https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
⚙️ Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

OpenCV / PIL

Matplotlib, Seaborn

Scikit-learn

Google Colab

KaggleHub

🧠 Model Architecture

The CNN model consists of:

Conv2D layers with ReLU activation

MaxPooling layers for spatial downsampling

Flatten layer

Fully connected Dense layers

Sigmoid output layer for binary classification

Training Configuration

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Evaluation Metric: Accuracy

🔄 Data Preprocessing & Augmentation
Preprocessing

Image resizing to 128 × 128

Normalization (pixel values scaled to [0, 1])

Removal of corrupted and non-image files

Data Augmentation

Rotation

Shear transformation

Zoom

Horizontal flipping

Nearest fill mode

📊 Exploratory Data Analysis

Random visualization of cat and dog images

Class distribution visualization using count plots

Dataset shuffling for unbiased training

📈 Results & Evaluation

Model trained on augmented image data

Evaluated using validation accuracy and loss

Performance visualized through:

Training vs Validation Accuracy

Training vs Validation Loss
