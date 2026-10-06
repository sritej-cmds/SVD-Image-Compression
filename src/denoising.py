import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_noise(image, noise_level=50, seed=None):
    """Add uniform random noise to a grayscale image.

    noise_level=50 produces noise approximately in the range [-25, 25].
    A seed can be supplied for reproducible results during testing.
    """
    if image.ndim != 2:
        raise ValueError("add_noise expects a grayscale 2D image")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    rng = np.random.default_rng(seed)
    noise = noise_level * (rng.random(image.shape) - 0.5)
    noisy = image.astype(np.float64) + noise

    return np.clip(noisy, 0, 255)


def denoise_svd(noisy_image, k):
    """Denoise an image using a rank-k truncated SVD reconstruction."""
    if noisy_image.ndim != 2:
        raise ValueError("denoise_svd expects a grayscale 2D image")
    if k <= 0:
        raise ValueError("k must be positive")

    U, S, Vt = np.linalg.svd(
        noisy_image.astype(np.float64),
        full_matrices=False
    )

    if k > len(S):
        raise ValueError(f"k={k} is larger than the available rank {len(S)}")

    # U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    # is the rank-k approximation U_k Sigma_k V_k^T.
    return np.einsum(
        "ik,k,kj->ij",
        U[:, :k],
        S[:k],
        Vt[:k, :]
    )


def save_image(image, output_path):
    """Clip an image to the valid grayscale range and save it as PNG."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    if not cv2.imwrite(output_path, image_uint8):
        raise IOError(f"Could not save image: {output_path}")


def save_comparison(original, noisy, denoised_results, output_dir):
    """Save original/noisy/denoised images and a comparison figure."""
    os.makedirs(output_dir, exist_ok=True)

    save_image(original, os.path.join(output_dir, "original.png"))
    save_image(noisy, os.path.join(output_dir, "noisy.png"))

    for k, image in denoised_results.items():
        save_image(image, os.path.join(output_dir, f"denoised_{k}.png"))

    # A single comparison figure is useful for the presentation/report.
    columns = 2 + len(denoised_results)
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Noisy")

    for axis, (k, image) in zip(axes[2:], denoised_results.items()):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"Denoised k={k}")

    for axis in axes:
        axis.axis("off")

    fig.suptitle("SVD Image Denoising")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "denoising_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    output_dir = "outputs/denoising"
    image = cv2.imread("images/checkers.pgm", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Could not load image: images/checkers.pgm")
    noisy = add_noise(image, noise_level=50, seed=42)
    results = {k: denoise_svd(noisy, k) for k in [10, 30, 50]}

    save_comparison(image, noisy, results, output_dir)

    print("===== SVD DENOISING =====")
    print("Image shape:", image.shape)
    print("Noise level: 50")
    print("Tested k values: 10, 30, 50")
    print(f"Outputs saved to: {output_dir}/")