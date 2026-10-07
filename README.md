# SVD-Image-Compression
# SVD Image Compression and Denoising

## Mini Project – Linear Algebra

### Application of Singular Value Decomposition in Image Processing

---

## 1. Project Overview

This project demonstrates how **Singular Value Decomposition (SVD)**, a fundamental concept in Linear Algebra, can be applied to real-world image-processing problems.

The project progresses from the mathematical understanding of SVD to practical applications:

1. SVD Fundamentals
2. Geometric Interpretation of SVD
3. Image Compression using SVD
4. Image Denoising using SVD

The central mathematical decomposition used throughout the project is:
A = U*Sigma*V^T

where:

- \(A\) is the original matrix
- \(U\) contains the left singular vectors
- \(\Sigma\) contains the singular values
- \(V^T\) is the transpose of the right singular-vector matrix

The project demonstrates how the singular values can be used to identify the most significant information in a matrix and how retaining only important components can produce a lower-rank approximation.

---

# 2. Project Objectives

The main objectives of this project are:

- Understand the mathematical structure of SVD.
- Verify the orthogonality of singular-vector matrices.
- Understand the geometric interpretation of SVD.
- Demonstrate the transformations represented by \(V^T\), \(\Sigma\), and \(U\).
- Apply SVD to a real-world image represented as a matrix.
- Perform low-rank image approximation.
- Demonstrate image compression using dominant singular components.
- Calculate the reduction in stored values after compression.
- Demonstrate noise reduction using SVD-based low-rank approximation.
- Connect Linear Algebra concepts with practical image-processing applications.

---

# 3. Mathematical Background

For a matrix \(A\), Singular Value Decomposition represents the matrix as:

\[
A = U\Sigma V^T
\]

The decomposition separates the transformation into three stages.

### \(V^T\) – Rotation / Reflection

The matrix \(V^T\) changes the orientation of the input data.

### \(\Sigma\) – Scaling

The diagonal matrix \(\Sigma\) scales the transformed data according to the singular values.

### \(U\) – Rotation / Reflection

The matrix \(U\) performs another orthogonal transformation to obtain the final result.

Therefore, the transformation can be viewed conceptually as:

```text
Original Data
      |
      v
     V^T
      |
      v
  Rotation /
  Reflection
      |
      v
     Σ
      |
      v
   Scaling
      |
      v
      U
      |
      v
  Rotation /
  Reflection
      |
      v
