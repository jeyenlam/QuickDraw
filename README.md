# QuickDraw

## Features
- **Real-time Hand Landmark Detection**: Detects hand landmarks in real-time using a webcam.
- **Canvas Drawing**: Allows users to draw on a canvas using their index finger.
- **Drawing Prediction**: Provides real-time predictions of the user's drawing based on pre-trained categories.

## Dataset
The dataset is sourced from Google's Quick, Draw! dataset, which contains millions of doodles across various categories. I used the simplified, preprocessed version, available [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&inv=1&invt=AbliDA)

## Training
I selected my preferred categories from the dataset and downloaded the corresponding files. Each category contains thousands of doodles, but I extracted 5,000 samples per category. With six categories, the total dataset comprises 30,000 samples.

For training, I used a **Multi-Layer Perceptron (MLP)** model, implemented in the **./training** directory.

## Model Accuracy

## Requirements
- opencv-python
- matplotlib
- numpy
- scikit-learn
- mediapipe
