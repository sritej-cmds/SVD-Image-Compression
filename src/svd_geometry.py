import numpy as np
import matplotlib.pyplot as plt

A=np.array([[2,1],[-1,1]],dtype=float)

U,S,Vt=np.linalg.svd(A)

t=np.linspace(0,2*np.pi,100)
X=np.array([np.cos(t),np.sin(t)])


# ============================================================
# TASK 3 — Vᵀ TRANSFORMATION
# ============================================================

def apply_vt(X,Vt):
    return Vt@X

VX=apply_vt(X,Vt)

plt.figure(figsize=(6,6))
plt.plot(X[0],X[1],label="Original Circle")
plt.plot(VX[0],VX[1],label="After Vᵀ")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Vᵀ Transformation")
plt.show()


# ============================================================
# TASK 4 — Σ TRANSFORMATION
# ============================================================

def apply_sigma(X,S):
    Sigma=np.diag(S)
    return Sigma@X

SVX=apply_sigma(VX,S)

plt.figure(figsize=(6,6))
plt.plot(VX[0],VX[1],label="After Vᵀ")
plt.plot(SVX[0],SVX[1],label="After ΣVᵀ")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Σ Transformation")
plt.show()
plt.close()

# ============================================================
# TASK 5 — U TRANSFORMATION
# ============================================================

def apply_u(X,U):
    return U@X

AX=apply_u(SVX,U)

plt.figure(figsize=(6,6))
plt.plot(SVX[0],SVX[1],label="After ΣVᵀ")
plt.plot(AX[0],AX[1],label="After UΣVᵀ")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("U Transformation")
plt.show()
plt.close()

# ============================================================
# TASK 6 — REFLECTION EXPERIMENT
# ============================================================

V=Vt.T

det_U=np.linalg.det(U)
det_V=np.linalg.det(V)

print("\n===== TASK 6 — REFLECTION EXPERIMENT =====")

print("det(U):",det_U)
print("det(V):",det_V)

if det_U<0:
    print("U contains a reflection.")
else:
    print("U does not contain a reflection.")

if det_V<0:
    print("V contains a reflection.")
else:
    print("V does not contain a reflection.")


U1=U.copy()
V1=V.copy()

U1[:,1]*=-1
V1[:,1]*=-1

A_modified=U1@np.diag(S)@V1.T

print("\nOriginal A:")
print(A)

print("\nModified U1:")
print(U1)

print("\nModified V1:")
print(V1)

print("\nU1ΣV1ᵀ:")
print(A_modified)

print("\nModified SVD reconstructs A:")
print(np.allclose(A_modified,A))

# ============================================================
# TASK 7 — NUMERICAL VERIFICATION : AV=U*sigma
# ============================================================

def verify_svd_geometry(A,U,S,Vt):
    left=A@Vt.T
    right=U@np.diag(S)
    difference=left-right
    return np.allclose(left,right),difference

check,difference=verify_svd_geometry(A,U,S,Vt)

print("\n===== TASK 7 — NUMERICAL VERIFICATION =====")

print("\nAV:")
print(A@Vt.T)

print("\nUΣ:")
print(U@np.diag(S))

print("\nDifference:")
print(difference)

print("\nAV=UΣ:",check)

# ============================================================
# FINAL — COMPLETE SVD GEOMETRY
# ============================================================

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot(X[0],X[1])
plt.axis("equal")
plt.grid(True)
plt.title("Original Unit Circle")

plt.subplot(2,2,2)
plt.plot(VX[0],VX[1])
plt.axis("equal")
plt.grid(True)
plt.title("After Vᵀ")

plt.subplot(2,2,3)
plt.plot(SVX[0],SVX[1])
plt.axis("equal")
plt.grid(True)
plt.title("After ΣVᵀ")

plt.subplot(2,2,4)
plt.plot(AX[0],AX[1])
plt.axis("equal")
plt.grid(True)
plt.title("After UΣVᵀ = AX")

plt.tight_layout()
plt.show()
plt.close()
