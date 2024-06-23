import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'

# 2 Jenis
amanita_path = 'data/Amanita_Caesarea-Edible'
gyromitra_path ='data/Gyromitra_Esculenta-NotEdible'


# 4 Jenis
# amanita_path = 'data/Amanita_Caesarea-Edible'
# boletus_path = 'data/Boletus_Regius-Edible'
# gyromitra_path ='data/Gyromitra_Esculenta-NotEdible'
# omphalotus_path ='data/Omphalotus_Olearius-NotEdible'

# Definisi kelas
classes = {
    # 2 Jenis
    0: 'Amanita_Caesarea-Edible',
    1: 'Gyromitra_Esculenta-NotEdible'


    # 4 Jenis
    # 0: 'Amanita_Caesarea-Edible',
    # 1: 'Boletus_Regius-Edible',
    # 2: 'Gyromitra_Esculenta-NotEdible',
    # 3: 'Omphalotus_Olearius-NotEdible'
}

# Preprocess dan augmentasi gambar
def preprocess_and_augment(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    # Augmentasi
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((128, 128), angle, 1)
    image = cv2.warpAffine(image, M, (256, 256))
    value = np.random.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image / 255.0
    return gray_image

# Menghitung fitur GLCM
def compute_glcm_features(image):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix((image * 255).astype(np.uint8), distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    features = np.hstack([contrast, correlation, energy, homogeneity])
    return features

# Load gambar dan label
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
# 2 Jenis
amanita_features, amanita_labels = load_images_and_labels(amanita_path, 0)
gyromitra_features, gyromitra_labels = load_images_and_labels(gyromitra_path, 1)


# 4 Jenis
# amanita_features, amanita_labels = load_images_and_labels(amanita_path, 0)
# boletus_features, boletus_labels = load_images_and_labels(boletus_path, 1)
# gyromitra_features, gyromitra_labels = load_images_and_labels(gyromitra_path, 2)
# omphalotus_features, omphalotus_labels = load_images_and_labels(omphalotus_path, 3)


# Combine data
# 2 Jenis
X = amanita_features + gyromitra_features
y = amanita_labels + gyromitra_labels


# 4 Jenis
# X = amanita_features + boletus_features + gyromitra_features + omphalotus_features
# y = amanita_labels + boletus_labels + gyromitra_labels + omphalotus_labels


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# Test
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(classes.values()), yticklabels=list(classes.values()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Training and Testing Accuracy
train_accuracy = classifier.score(X_train, y_train)
test_accuracy = classifier.score(X_test, y_test)
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'green'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.show()

# ROC Curve
from sklearn.preprocessing import label_binarize

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=[0, 1])
n_classes = y_test_binarized.shape[1]

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], classifier.predict_proba(X_test)[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarized[:, i], classifier.predict_proba(X_test)[:, i])

# Plot ROC curve
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Simpan model
joblib.dump(classifier, 'knn_model.pkl')
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

if __name__ == '_main_':
    app.run(debug=True)