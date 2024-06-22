import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'

# Folder paths
amanita_path = 'data/Amanita calyptroderma'
coprinellus_path = 'data/Coprinellus micaceus'
coprimopsis_path = 'data/Coprinopsis lagopus'
ganoderma_path = 'data/Ganoderma tsugae'
stereum_path = 'data/Stereum ostrea'
volvopluteus_path ='data/Volvopluteus gloiocephalus'

# Class labels
classes = {
    0: 'Amanita calyptroderma',
    1: 'Coprinellus micaceus',
    2: 'Coprinopsis lagopus',
    3: 'Ganoderma tsugae',
    4: 'Stereum ostrea',
    5: 'Volvopluteus gloiocephalus'
}

# Function to preprocess and augment images
def preprocess_and_augment(image_path):
    # Read image
    image = cv2.imread(image_path)
    # Resize image to a standard size
    image = cv2.resize(image, (256, 256))

    # Augmentation: random flip, rotation, and brightness adjustment
    # Horizontal flip
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    # Rotation
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((128, 128), angle, 1)
    image = cv2.warpAffine(image, M, (256, 256))
    # Brightness adjustment
    value = np.random.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize image
    gray_image = gray_image / 255.0
    return gray_image

# Function to compute GLCM features
def compute_glcm_features(image):
    distances = [1]  # Pixel distance
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Directions 0, 45, 90, 135 degrees
    glcm = graycomatrix((image * 255).astype(np.uint8), distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    features = np.hstack([contrast, correlation, energy, homogeneity])
    return features

# Load images and compute features
def load_images_and_labels(path, label):
    features = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(path, filename)
            preprocessed_image = preprocess_and_augment(filepath)
            image_features = compute_glcm_features(preprocessed_image)
            features.append(image_features)
            labels.append(label)
    return features, labels

# Load data
amanita_features, amanita_labels = load_images_and_labels(amanita_path, 0)
coprinellus_features, coprinellus_labels = load_images_and_labels(coprinellus_path, 1)
coprimopsis_features, coprimopsis_labels = load_images_and_labels(coprimopsis_path, 2)
ganoderma_features, ganoderma_labels = load_images_and_labels(ganoderma_path, 3)
stereum_features, stereum_labels = load_images_and_labels(stereum_path, 4)
volvopluteus_features, volvopluteus_labels = load_images_and_labels(volvopluteus_path, 5)

# Combine data
X = amanita_features + coprinellus_features + coprimopsis_features + ganoderma_features + stereum_features + volvopluteus_features
y = amanita_labels + coprinellus_labels + coprimopsis_labels + ganoderma_labels + stereum_labels + volvopluteus_labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN classifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# Test classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(classifier, 'knn_model.pkl')

# Load the trained model
classifier = joblib.load('knn_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            preprocessed_image = preprocess_and_augment(file_path)
            features = compute_glcm_features(preprocessed_image)
            prediction = classifier.predict([features])[0]
            probabilities = classifier.predict_proba([features])[0]
            knn_probabilities = [(classes[i], prob) for i, prob in enumerate(probabilities)]
            result = classes[prediction]
            return render_template('index.html', result=result, features=features, knn_probabilities=knn_probabilities, accuracy=accuracy)
    return render_template('index.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
