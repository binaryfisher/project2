SmartHomeGestureControl Part 2

Overview
This project implements a gesture recognition system designed to identify specific hand gestures from video files. Utilizing advanced machine learning techniques and computer vision, the system extracts features from a selected frame within each video, compares these features against a pre-trained model, and identifies the gesture based on the highest similarity. 

Features
Feature extraction from video frames using OpenCV.
Gesture recognition using TensorFlow for calculating cosine similarity between extracted features and pre-trained models.
Output generation in CSV format listing recognized gestures alongside their corresponding video files.

Setup Instructions
Prerequisites
Python 3.x
OpenCV library (cv2)
TensorFlow
NumPy