import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Data
airplane_images = np.load('./data/full_numpy_bitmap_airplane.npy')
angel_images = np.load('./data/full_numpy_bitmap_angel.npy')
banana_images = np.load('./data/full_numpy_bitmap_banana.npy')
bowtie_images = np.load('./data/full_numpy_bitmap_bowtie.npy')
butterfly_images = np.load('./data/full_numpy_bitmap_butterfly.npy')
cactus_images = np.load('./data/full_numpy_bitmap_cactus.npy')

# Limit each class to 1000 samples (6000 total)
airplane_images = airplane_images[:1000]
angel_images = angel_images[:1000]
banana_images = banana_images[:1000]
bowtie_images = bowtie_images[:1000]
butterfly_images = butterfly_images[:1000]
cactus_images = cactus_images[:1000]

# Concatenate the images into one dataset
X = np.concatenate([airplane_images, angel_images, banana_images, bowtie_images, butterfly_images, cactus_images], axis=0)

airplane_labels = np.zeros(airplane_images.shape[0], dtype=int)  # 0 for airplane
angel_labels = np.ones(angel_images.shape[0], dtype=int)  # 1 for angel
banana_labels = np.full(banana_images.shape[0], 2, dtype=int)  # 2 for banana
bowtie_labels = np.full(bowtie_images.shape[0], 3, dtype=int)  # 3 for bowtie
butterfly_labels = np.full(butterfly_images.shape[0], 4, dtype=int)  # 4 for butterfly
cactus_labels = np.full(cactus_images.shape[0], 5, dtype=int)  # 5 for cactus

# Concatenate labels into one array
y = np.concatenate([airplane_labels, angel_labels, banana_labels, bowtie_labels, butterfly_labels, cactus_labels], axis=0)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the data for MLP input
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Create and train MLP model using scikit-learn
model = MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', solver='adam', max_iter=10)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, target_names=['airplane', 'angel', 'banana', 'bowtie', 'butterfly', 'cactus']))

save_dir = './src'
model_path = os.path.join(save_dir, 'quickdraw_mlp_model.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)