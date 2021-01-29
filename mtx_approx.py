import numpy as np


'''
@params:
    A: m × n matrix
    B: n × p matrix
    c: a positive integer
    ps: a list of probabilities (size n)
@return:
    Matrix C and R such that CR ≈ AB
    C: m × c matrix
    R: c × p matrix
'''
def matrix_multi_approx(A: np.array, B: np.array, c: int):
    assert A.shape[1] == B.shape[0], "Shape not match"
    n = A.shape[1]
    probs = compute_ps(A, B)
    C = np.zeros((A.shape[0], c))
    R = np.zeros((c, B.shape[1]))
    for i in range(c):
        it = np.random.choice(n, p=probs)
        C[:, i] = A[:, it]/np.sqrt(c*probs[it])
        R[i, :] = B[it, :]/np.sqrt(c*probs[it])
    return C, R


'''
@params:
    A: m × n matrix
    B: n × p matrix
@return:
    a list of probabilities that minimize E(||AB-CR||^2) (Frobenius norm)
'''
def compute_ps(A: np.array, B: np.array):
    assert A.shape[1] == B.shape[0], "Shape not match"
    n = A.shape[1]
    C =  np.sum(np.array(
        [np.linalg.norm(A[:, k],2) * np.linalg.norm(B[k, :],2) for k in range(n)]
    ))
    return \
        np.array(
            [np.linalg.norm(A[:, k],2) * np.linalg.norm(B[k, :], 2) / C for k in range(n)]
        )


# Construct large scale matrix
x = np.arange(-0.7, 0.701, 0.001)
y = np.arange(-0.7, 0.701, 0.001)
xx, yy = np.meshgrid(x, y, sparse=False)
# calculate z for each x_i and y_j, and store in matrix A
A = B = np.sqrt(1 - xx**2 - yy**3)

# Do approximation
C, R = matrix_multi_approx(A, B, 500)

# Check the Frobenius norm of AB-CR
AB = A@B
CR = C@R
fro_norm = np.linalg.norm(AB-CR, 'fro')

