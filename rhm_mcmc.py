import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import sympy as sm


def H(theta, p, logpdf, metric, metric_inv, dim=4):
    H = -logpdf()
    H += 0.5 * np.log((2 * np.pi) ** dim * np.linalg.det(metric(theta)))
    H += 0.5 * np.dot(p.T, np.dot(metric_inv(theta), p))
    return H


def dHdp(theta, p, metric_inv):
    return np.dot(metric_inv(theta), p)


def dHdtheta(theta, p, logpdf, metric_inv, dGdtheta):
    dH = logpdf(theta)
    dH -= 0.5 * np.trace(metric_inv(theta) @ dGdtheta, axis1=1, axis2=2)
    dH += 0.5 * np.linalg.multi_dot([p.T, metric_inv(theta), dGdtheta, metric_inv(theta), p])
    return -dH


def leapfrog(theta, p, logpdf, metric, metric_inv, dGdtheta, epsilon):
    p_e2 = scipy.optimize.fixed_point(lambda p_e2: p - epsilon / 2 * dHdtheta(theta, p_e2, logpdf, metric, metric_inv),
                                      p)
    theta_e = scipy.optimize.fixed_point(lambda theta_e:
                                         theta - epsilon / 2 * (
                                                     dHdp(theta, p_e2, metric_inv) + dHdtheta(theta_e, p_e2, logpdf,
                                                                                              metric_inv, dGdtheta)),
                                         theta)
    p_e = p_e2 - epsilon / 2 * dHdtheta(theta_e, p_e2, logpdf, metric, metric_inv)
    return theta_e, p_e


if __name__ == '__main__':
    import autograd.numpy as np
    from autograd import hessian

    def f(x):
        y = np.exp(x)
        y = np.sum(y)
        return y

    xx = np.ones(4)
    ddf = hessian(f)

    print(ddf(xx).shape)
    print(ddf(xx))
    print(ddf(xx).diagonal())

    def mhessian(f):
        ddf = hessian(f)
        def hess(x):
            return ddf(x).diagonal(axis1=1, axis2=2)
        return hess
