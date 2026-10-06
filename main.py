import os
import numpy as np

from src.svd_basics import compute_svd, verify_orthogonality
from src.svd_geometry import run_geometry_experiment
from src.compression import (
    load_image,
    image_svd,
    plot_singular_values,
    compress_image,
    save_compressed_image,
    compression_percentage,
)
from src.denoising import add_noise, denoise_svd, save_comparison


ROOT = os.path.dirname(os.path.abspath(__file__))


def path(*parts):
    return os.path.join(ROOT, *parts)


def run_svd_basics():
    print("\n" + "=" * 60)
    print("PART 1 — SVD FUNDAMENTALS")
    print("=" * 60)

    A = np.array([
        [2, 1],
        [-1, 1]
    ], dtype=float)

    U, S, Vt = compute_svd(A)

    print("Matrix A:\n", A)
    print("\nU:\n", U)
    print("\nSingular values:\n", S)
    print("\nVt (Vᵀ):\n", Vt)
    print("\nOrthogonality verified:", verify_orthogonality(U, Vt))

    return A, U, S, Vt


def run_geometry():
    print("\n" + "=" * 60)
    print("PART 2 — SVD GEOMETRY")
    print("=" * 60)

    result = run_geometry_experiment(path("outputs", "geometry"))
    print("det(U):", result["det_U"])
    print("det(V):", result["det_V"])
    print("Modified SVD reconstructs A:", result["reflection_reconstruction"])
    print("AV = UΣ:", result["av_equals_usigma"])
    print("Geometry outputs saved to outputs/geometry/")


def run_compression():
    print("\n" + "=" * 60)
    print("PART 3 — IMAGE COMPRESSION")
    print("=" * 60)

    image = load_image(path("images", "einstein.jpg"))
    U, S, Vt = image_svd(image)

    print("Image shape:", image.shape)
    print("SVD shapes:")
    print("  U:", U.shape)
    print("  S:", S.shape)
    print("  Vt:", Vt.shape)

    output_dir = path("outputs", "compression")
    os.makedirs(output_dir, exist_ok=True)
    plot_singular_values(S, os.path.join(output_dir, "singular_values.png"))
    save_compressed_image(image, os.path.join(output_dir, "original.png"))

    m, n = image.shape
    for k in [50, 100, 150]:
        compressed = compress_image(U, S, Vt, k)
        output_path = os.path.join(output_dir, f"einstein_k{k}.png")
        save_compressed_image(compressed, output_path)
        percentage = compression_percentage(m, n, k)
        print(f"k={k}: {percentage:.2f}% storage reduction -> {output_path}")


def run_denoising():
    print("\n" + "=" * 60)
    print("PART 4 — IMAGE DENOISING")
    print("=" * 60)

    image = load_image(path("images", "checkers.pgm"))
    noisy = add_noise(image, noise_level=50, seed=42)
    results = {}

    for k in [10, 30, 50]:
        results[k] = denoise_svd(noisy, k)
        print(f"Denoising completed for k={k}")

    output_dir = path("outputs", "denoising")
    save_comparison(image, noisy, results, output_dir)
    print("Denoising outputs saved to outputs/denoising/")


def main():
    print("\n===== SVD IMAGE COMPRESSION PROJECT =====")
    run_svd_basics()
    run_geometry()
    run_compression()
    run_denoising()
    print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
