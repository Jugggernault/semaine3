import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image
import matplotlib.pyplot as plt


def load_and_preprocess(image_path):
    """Loads the image, converts to RGB, and flattens for clustering."""
    try:
        print("Load and preprocess image...")
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        flat_image = img_np.reshape((-1, 3))
        flat_image = np.float32(flat_image)
        return img_np, flat_image
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None


def perform_mean_shift(flat_image, bandwidth=None, quantile=0.2, n_samples=500):
    """Performs Mean Shift clustering."""
    if bandwidth is None:
        bandwidth = estimate_bandwidth(
            flat_image, quantile=quantile, n_samples=n_samples
        )

    print("Performing mean shift")
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    return labels, cluster_centers


def reconstruct_image(img_np, labels, cluster_centers):
    """Reconstructs the image from cluster labels and centers."""
    segmented_image = cluster_centers[labels].astype(np.uint8).reshape(img_np.shape)
    return segmented_image


def visualize_segmentation(original_image, segmented_image):
    """Displays the original and segmented images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[1].imshow(segmented_image)
    axes[1].set_title("Segmented Image")
    axes[0].set_axis_off()
    axes[1].set_axis_off()
    plt.show()


def mean_shift_segmentation(image_path, bandwidth=None, quantile=0.2, n_samples=500):
    """Main function for Mean Shift segmentation."""
    img_np, flat_image = load_and_preprocess(image_path)
    if img_np is None:
        return None
    labels, cluster_centers = perform_mean_shift(
        flat_image, bandwidth, quantile, n_samples
    )
    segmented_image = reconstruct_image(img_np, labels, cluster_centers)
    return img_np, segmented_image


# Example usage:
image_path = "cat.jpg"  # Replace with your image path
original_img, segmented_img = mean_shift_segmentation(
    image_path, quantile=0.1, bandwidth=50
)

if segmented_img is not None:
    visualize_segmentation(original_img, segmented_img)
