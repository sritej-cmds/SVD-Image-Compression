import numpy as np
import matplotlib.pyplot as plt
import os


def create_unit_circle():
    t=np.linspace(0,2*np.pi,100)
    X=np.array([np.cos(t),np.sin(t)])

    e1=np.array([1,0])
    e2=np.array([0,1])

    plt.figure(figsize=(6,6))

    plt.plot(X[0],X[1],label="Unit Circle")

    plt.quiver(0,0,e1[0],e1[1],angles='xy',scale_units='xy',scale=1,label="e1")
    plt.quiver(0,0,e2[0],e2[1],angles='xy',scale_units='xy',scale=1,label="e2")

    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

    plt.axhline(0)
    plt.axvline(0)

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Unit Circle")

    os.makedirs("outputs/geometry",exist_ok=True)
    plt.savefig("outputs/geometry/task1_unit_circle.png")
    plt.show()

    return X


def compute_svd(A):
    return np.linalg.svd(A)


def verify_orthogonality(U,Vt):
    u_check=U.T@U
    v_check=Vt@Vt.T

    print("U.T @ U:")
    print(u_check)

    print("\nVt @ Vt.T:")
    print(v_check)

    u_identity=np.eye(U.shape[1])
    v_identity=np.eye(Vt.shape[0])

    return np.allclose(u_check,u_identity) and np.allclose(v_check,v_identity)


if __name__=="__main__":

    A=np.array([
        [2,1],
        [-1,1]
    ])

    print("===== SVD FUNDAMENTALS =====")

    print("\nMatrix A:")
    print(A)

    U,S,Vt=compute_svd(A)

    print("\nU:")
    print(U)

    print("\nSingular values:")
    print(S)

    print("\nVt (V^T):")
    print(Vt)

    print("\nOrthogonality Check:")
    result=verify_orthogonality(U,Vt)

    print("\nAre U and V orthogonal?")
    print(result)

    print("\nGenerating unit circle...")
    create_unit_circle()

    print("\nUnit circle saved to:")
    print("outputs/geometry/task1_unit_circle.png")