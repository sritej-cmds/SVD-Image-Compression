import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image


def image_svd(image):
    return np.linalg.svd(
        image.astype(np.float64),
        full_matrices=False
    )


def plot_singular_values(S):
    plt.figure(figsize=(10, 5))

    plt.plot(S)

    plt.title("Singular Values of Einstein Image")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "outputs/compression/singular_values.png"
    )

    plt.close()


def compress_image(U, S, Vt, k):
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    return np.einsum(
        "ik,k,kj->ij",
        U_k,
        S_k,
        Vt_k
    )


def save_compressed_image(image, output_path):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, image)


def compression_percentage(m, n, k):
    original_values = m * n
    compressed_values = m * k + k + k * n

    return (1 - compressed_values / original_values) * 100

def get_compression_stats(m, n, k):
    original_values = m * n
    compressed_values = m * k + k + k * n
    reduction = compression_percentage(m, n, k)

    return {
        "k": k,
        "original_values": original_values,
        "compressed_values": compressed_values,
        "storage_reduction": reduction
    }


if __name__ == "__main__":

    image = load_image("images/einstein.jpg")

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    U, S, Vt = image_svd(image)

    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("Vt shape:", Vt.shape)

    print("\nFirst 10 singular values:")
    print(S[:10])

    # Plot singular values
    plot_singular_values(S)

    # Compress using different values of k
    for k in [50, 100, 150]:

        compressed = compress_image(U, S, Vt, k)

        compressed_image = np.clip(
            compressed,
            0,
            255
        ).astype(np.uint8)

        output_path = (
            f"outputs/compression/einstein_k{k}.png"
        )

        save_compressed_image(
            compressed_image,
            output_path
        )

        print(
            f"\nk = {k}"
            f"\nCompressed shape: {compressed.shape}"
            f"\nMinimum value: {compressed.min():.2f}"
            f"\nMaximum value: {compressed.max():.2f}"
            f"\nSaved: {output_path}"
        )

    # Compression statistics
    m, n = image.shape

    print("\n--- Compression Results ---")

    for k in [50, 100, 150]:

        original_values = m * n
        compressed_values = m * k + k + k * n

        percentage = compression_percentage(m, n, k)

        print(f"\nk = {k}")
        print(f"Original values: {original_values:,}")
        print(f"Compressed values: {compressed_values:,}")
        print(f"Storage reduction: {percentage:.2f}%")