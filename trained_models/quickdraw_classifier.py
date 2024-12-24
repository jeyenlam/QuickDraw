import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Dataset
angel_images = np.load('../data/full_numpy_bitmap_angel.npy')
banana_images = np.load('../data/full_numpy_bitmap_banana.npy')

# Limit the dataset to 2000 samples
angel_images = angel_images[:1000]
banana_images = banana_images[:1000]

X = np.concatenate([angel_images, banana_images], axis=0)
angel_labels = np.zeros(angel_images.shape[0], dtype=int)  # 0 for angel
banana_labels = np.ones(banana_images.shape[0], dtype=int)  # 1 for banana
y = np.concatenate([angel_labels, banana_labels], axis=0)

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
print(classification_report(y_test, y_pred, target_names=['angel', 'banana']))

save_dir = './src'
model_path = os.path.join(save_dir, 'quickdraw_mlp_model.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)