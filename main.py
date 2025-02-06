import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

# Define paths
image_folder = r"C:\Users\vasan\Transformers\Image"
label_file = r"C:\Users\vasan\Transformers\Labels\Labels.csv"

# Load labels
df = pd.read_csv(label_file, delimiter=",")  # Read CSV with comma as delimiter
df.columns = ["Filename", "Label"]  # Rename columns
#df["Label"] = df["Label"].str.strip()  # Remove leading/trailing whitespaces
labels = df["Label"].values
image_names = df["Filename"].values

print("Number of Labels:", len(labels), "Number of Images:", len(image_names))

# Load images
images = []
valid_labels = []

for img_name, label in zip(image_names, labels):
    img_name += ".jpg"  # Append .jpg to the image name
    img_path = os.path.normpath(os.path.join(image_folder, img_name))  # Normalize path

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Image {img_name} could not be loaded.")
        continue  
    img = cv2.resize(img, (224, 224))  # Resize images
    img = img / 255.0  # Normalize pixel values
    images.append(img)
    valid_labels.append(label)  # Keep only labels of valid images

images = np.array(images)
valid_labels = np.array(valid_labels)  # Convert valid_labels to numpy array
print("Number of successfully loaded images:", len(images))

# Encode labels
unique_labels = sorted(set(valid_labels))
label_map = {label: i for i, label in enumerate(unique_labels)}
numeric_labels = np.array([label_map[label] for label in valid_labels])
encoded_labels = to_categorical(numeric_labels)

# Ensure the number of images matches the number of labels
if len(images) != len(encoded_labels):
    raise ValueError("The number of images does not match the number of labels.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

print("Data preprocessing complete. Training samples:", len(X_train), "Test samples:", len(X_test))