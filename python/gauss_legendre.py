"""
The Gauss-Legendre quadrature formula is an accurate method for numerical integration,
particularly for smooth functions
https://en.wikipedia.org/wiki/Gauss–Legendre_quadrature
"""

import numpy as np


def gauss_legendre(n, a, b):
    """
    Computes Gauss-Legendre weights and roots for an arbitrary interval [a,b]

    Parameters:
    ----------
    N : int
        The number of quadrature points
    a : float
        The lower bound of the interval
    b : float
        The upper bound of the interval

    Returns:
    -------
    x : numpy.ndarray
        An array containing the roots of the Gauss-Legendre quadrature formula
    w : numpy.ndarray
        An array containing the weights of the Gauss-Legendre quadrature formula

          Examples:
    --------
    >>> x, w = gauss_legendre(2, 0, 1)
    >>> np.allclose(x, np.array([0.21132487,  0.78867513]))
    True
    >>> np.allclose(w, np.array([0.50000000, 0.500000]))
    True
    >>> x, w = gauss_legendre(5, 1, 3)
    >>> np.allclose(x, np.array([1.09382015, 1.46153069, 2.        , 2.53846931, 2.90617985]))
    True
    >>> np.allclose(w, np.array([0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]))
    True
    """

    # Ensure that a < b
    assert a < b

    M = np.zeros((n, n))
    for i in range(n):
        if i < n - 1:
            M[i, i + 1] = np.sqrt(1 / (4 - (i + 1) ** -2))

    M_ = M + M.T
    x, V = np.linalg.eigh(M_)

    # Scale eigenvalues to lie in [a,b] and compute weights
    w = np.zeros(n)
    assert w.size == x.size
    for i in range(n):
        w[i] = 0.5 * 2.0 * (b - a) * V[0, i] ** 2
        x[i] = (b - a) * 0.5 * x[i] + (b + a) * 0.5

    return x, w


# print(gauss_legendre(5, 1, 3))

if __name__ == "__main__":
    from doctest import testmod

    testmod()
