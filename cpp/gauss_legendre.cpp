/**
* @file gauss_legendre.cpp
* @brief Computes Gauss-Legendre weights and roots for an arbitrary interval [a,b]
* @author Ajay Melekamburath
* @date 22 April 2023
*/

#include "Eigen/Dense"
#include <iostream>

/**
 * @brief Computes Gauss-Legendre weights and roots for an arbitrary interval [a,b]
 * @param n The number of quadrature points
 * @param a The lower bound of the interval
 * @param b The upper bound of the interval
 * @return A pair of Eigen vectors, the first containing the roots and the second containing the weights
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd> gauss_legendre(int n, double a,
                                                           double b);

/**
 * This function computes the Gauss-Legendre quadrature formula for an arbitrary interval [a,b] and returns the weights and roots
 * as two Eigen vectors. The Gauss-Legendre quadrature formula is an accurate method for numerical integration, particularly
 * for smooth functions.
 * https://en.wikipedia.org/wiki/Gauss–Legendre_quadrature
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd>
gauss_legendre(const int n, const double a, const double b) {
    // make sure a < b
    assert(a < b && "a should be less than b");
    Eigen::MatrixXd M(n, n);
    M.fill(0.0);

    for (auto i = 0; i < n; i++) {
        if (i < n - 1) { M(i, i + 1) = sqrt(1 / (4 - pow(i + 1, -2))); }
    }
    const Eigen::MatrixXd M_ = M + M.transpose();

    // compute eigenvalues and eigenvectors
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M_);
    const Eigen::VectorXd evals = solver.eigenvalues();
    const Eigen::MatrixXd evecs = solver.eigenvectors();

    Eigen::VectorXd w = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    // scale the computed values
    for (auto i = 0; i < n; i++) {
        w(i) = 0.5 * 2.0 * (b - a) * evecs(0, i) * evecs(0, i);
        x(i) = (b - a) * 0.5 * evals(i) + (b + a) * 0.5;
    }

    return std::make_pair(x, w);
}

int main() {
    auto [x, w] = gauss_legendre(5, 1.0, 3.0);

    std::cout << "x: " << x.transpose() << std::endl;
    std::cout << "w: " << w.transpose() << std::endl;
    return 0;
}
