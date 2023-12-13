TensorFlow will be used to implement the solution for the autoencoder.
Just import everything and tweak the layers depending want you it to train on.
Was made in saturn 3.
Heres the orginal code just in case.///

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Assuming you have a dataset with features and labels, where 1 indicates fraud and 0 indicates normal.
# Load your dataset or replace this with your own data loading logic.
# Here, I'm using a hypothetical dataset for demonstration purposes.

# Load the dataset (replace 'your_dataset.csv' with the actual path)
dataset = pd.read_csv('your_dataset.csv')

# Separate features and labels
features = dataset.drop('label', axis=1)
labels = dataset['label']

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_standardized, labels, test_size=0.2, random_state=42)

# Build the autoencoder model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(X_train.shape[1], activation='linear')  # Output layer with linear activation
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Mean Squared Error is used as the loss function

# Train the model
model.fit(X_train, X_train, epochs=10, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate the mean squared error for each sample
mse = tf.keras.losses.mean_squared_error(X_test, predictions)
threshold = 0.01  # Adjust the threshold based on your specific scenario

# Identify anomalies based on the threshold
anomalies = mse > threshold

# Evaluate the model
accuracy = tf.reduce_mean(tf.cast(tf.equal(anomalies, y_test.values), dtype=tf.float32)).numpy()
print(f'Accuracy: {accuracy * 100:.2f}%')
