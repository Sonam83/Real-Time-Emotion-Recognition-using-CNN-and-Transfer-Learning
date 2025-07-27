# Real-Time-Emotion-Recognition-using-CNN-and-Transfer-Learning

About : This project implements a real time emotion recognition system using Convolution Neural Networks(CNN) and Transfer Learning with pretrained models like ResNet18 from keras applications

Procedure :

Environment check using Anaconda prompt in Jupyter and Google colab
Importing all the required Libraries
Creating a dataset along with sub-folders classification as ["Angry","Sad","Sleep","Smile","Surprise"] using Webcam/Iriun Webcam/Video path in jupyter with the help of tensor flow
Preprocessing involves Data Augmentation techniques like resizing, rescaling, normalisation adhered by splitting the data into training dataset and testing dataset using pytorch datasets
Creating batches using pytorch dataloaders
Building model with transfer learning
Train the model
saving the best model
Evaluating the model with metrics confusion matrix, precision, Recall, F1 score, Accuracy
Plotting the metrics plot on classification reports
Live stream capturing video with opencv haarcascade to predict the class : Inference using pytorch
Tech-stack used :

Python 3.11.13
Tensorflow 2.19.0
Opencv
Pytorch
Numpy,Matplotlib,Keras
Summary :

The best model parameters 'base_model': 'ResNet18', 'lr':0.001, 'freeze_all': True,  'activation': 'relu'}

The best training accuracy value is 91% and testing accuracy is 94%

Training is stable , avoiding overfitting

The model learns performs well on both training and validation data. -Final metrics: val_accuracy = 0.9978, accuracy = 0.9990

Acknowledgement : I would like to acknowledge the open source libraries Opencv,Tensorflow and Keras which played a significant role to streamline this face recognition project
