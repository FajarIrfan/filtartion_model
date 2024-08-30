import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the paths to your filtered dataset containing cat and dog images
cat_path = "/path/to/filtered/cats"
dog_path = "/path/to/filtered/dogs"

# Create lists to store images and labels
images = []
labels = []

# Load cat images and assign labels
for image_filename in os.listdir(cat_path):
    image_path = os.path.join(cat_path, image_filename)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize image to a consistent size
    images.append(image)
    labels.append("cat")

# Load dog images and assign labels
for image_filename in os.listdir(dog_path):
    image_path = os.path.join(dog_path, image_filename)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize image to a consistent size
    images.append(image)
    labels.append("dog")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Flatten the images for input to the classifier
images_flat = images.reshape(images.shape[0], -1)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

# Create and train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_images, train_labels)

# Make predictions on the test set
predictions = clf.predict(test_images)

# Calculate and print the accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)
