import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


def load_and_process(path):
    # Charger l'image avec OpenCV
    image = cv.imread(path)
    assert image is not None, "Image not found, check the file path and try again"

    # Convertir l'image de BGR à RGB pour un traitement cohérent
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Normaliser l'image entre 0 et 1 pour KMeans
    image = image / 255.0
    return image


def threshSegmentation(image):
    # Convertir en niveaux de gris
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Segmentation par seuillage
    _, tresh_image = cv.threshold(
        gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    return tresh_image


def clusteringSegmentation(image: np.ndarray, model):
    # Redimensionner l'image en tableau 2D pour KMeans
    pixels = image.reshape(-1, 3)

    # Appliquer le clustering KMeans
    model.fit(pixels)

    # Obtenir les centres des clusters (les couleurs dominantes)
    centers = model.cluster_centers_
    print("Centers: ", centers)
    print("Shape of centers: ", centers.shape)

    # Obtenir les labels des pixels
    labels = model.labels_
    print("Labels: ", labels)

    # Reconstruire l'image segmentée
    segmented_image = centers[labels].reshape(image.shape)
    return segmented_image


# Initialiser le modèle KMeans
kmean = KMeans(n_clusters=2, random_state=32)

# Charger et prétraiter l'image
image = load_and_process("cat.jpg")

# Appliquer la segmentation par clustering
new_image = clusteringSegmentation(image, kmean)

# Afficher l'image originale et segmentée
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Image originale
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image originale")
plt.axis("off")

# Image segmentée
plt.subplot(1, 2, 2)
plt.imshow(new_image)
plt.title("Image segmentée (KMeans)")
plt.axis("off")

plt.show()
