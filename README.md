# Email Fraud Detection Project

## Overview
This project aims to compare two machine learning models—one built using **TensorFlow** and the other using **Scikit-Learn**—for detecting fraudulent emails. We'll explore their differences, use cases, and performance metrics by researching and analyzing their perfomances.

## Data
The dataset used in this project contains labeled emails, where each email is marked as either fraudulent or not. The text data is one-hot encoded to transform it into a format suitable for the machine learning models.

## Models

### 1. TensorFlow Model
- **Purpose**: TensorFlow is an open-source framework primarily focused on **neural networks** and deep learning.
- **Architecture**: We build a neural network using TensorFlow's Keras API. The model consists of dense layers with dropout regularization to prevent overfitting.
- **Metrics**: We evaluate the model using accuracy, precision, recall, and a confusion matrix.

### 2. Scikit-Learn Model
- **Purpose**: Scikit-Learn is an open-source Python library for creating and evaluating machine learning models.
- **Features**: Includes a variety of supervised and unsupervised learning techniques (e.g., SVM, Random Forests, K-means).
- **Evaluation**: We assess the model using similar metrics as the TensorFlow model.

## Usage
To run the models and compare their performance, execute the respective Python scripts: `tensorflow_fraud_detection.py` and `scikit_fraud_detection.py`.

## Data Source

The dataset used in this project is sourced from the following Kaggle dataset:

- **Fraud Email Dataset**: This dataset contains the body of emails with their categories, which can be used for supervised learning. It is a subset of the dataset available at Radev, D. (2008), CLAIR collection of fraud email, ACL Data and Code Repository, ADCR2008T001.
Link to the dataset - https://www.kaggle.com/datasets/llabhishekll/fraud-email-dataset?resource=download

Please note that if you use this collection of fraud email in your research, you should include the above citation in any resulting papers.


## Requirements
Make sure you have the following libraries installed:
- TensorFlow
- Scikit-Learn
- Pandas
- Matplotlib

Install them using:
```sh
pip install tensorflow scikit-learn pandas matplotlib
```