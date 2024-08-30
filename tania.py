import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the path to your dataset containing cat and dog images
dataset_path ="C:\Users\ZEE LINKS\Desktop\codes\cats_and_dogs_filtered\validation\cats"

# Create lists to store images and labels
images = []
labels = []

# Loop through the dataset directory to load images and assign labels
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        label = 0 if category == "cat" else 1
        for image_filename in os.listdir(category_path):
            image_path = os.path.join(category_path, image_filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (100, 100))  # Resize image to a consistent size
            images.append(image)
            labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Flatten the images for input to the classifier
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Create and train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_images_flat, train_labels)

# Make predictions on the test set
predictions = clf.predict(test_images_flat)

# Calculate and print the accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)

