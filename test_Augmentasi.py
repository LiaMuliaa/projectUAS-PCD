import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_and_augment(image):
    # Resize image to a standard size
    image = cv2.resize(image, (256, 256))

    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Horizontal flip
    if np.random.rand() < 0.5:
        flipped_image = cv2.flip(image, 1)
        augmented_images.append(flipped_image)

    # Rotation
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((128, 128), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (256, 256))
    augmented_images.append(rotated_image)

    # Brightness adjustment
    value = np.random.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
    brightness_adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(brightness_adjusted_image)

    return augmented_images

# Load an example image
image_path = 'data\Amanita_Caesarea-Edible\Amanita_Caesarea_0011.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# Generate augmented images
augmented_images = preprocess_and_augment(image)

# Display the original and augmented images
titles = ['Original', 'Flipped', 'Rotated', 'Brightness Adjusted']
plt.figure(figsize=(10, 10))
for i, aug_image in enumerate(augmented_images):
    plt.subplot(2, 2, i+1)
    plt.imshow(cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.show()
