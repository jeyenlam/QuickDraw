<h1 align="center">QuickDraw</h1>

<div align="center">

  <p> A Python-based project inspired by Google's Quick, Draw game, where users can doodle with their index finger, and the drawings are predicted by a machine learning model.</p>
  <br/>
  
  ![Static Badge](https://img.shields.io/badge/project_status-in%20process-blue)

</div>

## Features
✅**Real-time Hand Landmark Detection**: Detects hand landmarks in real-time using a webcam.

✅**Canvas Drawing**: Allows users to draw on a canvas using their index finger.

✅**Drawing Prediction**: Provides predictions of the user's drawing based on pre-trained categories.

⬜ **Real-time Prediction**: Provides real-time predictions of the user's drawing based on pre-trained categories.

## Dataset
The dataset is sourced from Google's Quick, Draw! dataset, which contains millions of doodles across various categories. I used the simplified, preprocessed version, available [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&inv=1&invt=AbliDA)

## Training
I selected my preferred categories from the dataset and downloaded the corresponding files. Each category contains thousands of doodles, but I extracted 5,000 samples per category. With six categories, the total dataset comprises 30,000 samples.

For training, I used a **Multi-Layer Perceptron (MLP)** model, implemented in the **./training** directory.

## Model Accuracy

## Future Improvements

- **Web-based project**: Turn this project into a web app

## Requirements
- opencv-python
- matplotlib
- numpy
- scikit-learn
- mediapipe
