import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import sympy as sm
import time

from banana import Banana


def H(theta, p, logpdf, metric, metric_inv):
    dim = np.asarray(theta).shape[0]
    H = -logpdf(theta)
    H += 0.5 * np.log((2 * np.pi) ** dim * np.linalg.det(metric(theta)))
    H += 0.5 * np.dot(p.T, np.dot(metric_inv(theta), p))
    return H


def dHdp(theta, p, metric_inv):
    return np.dot(metric_inv(theta), p)


def dHdtheta(theta, p, grad_logpdf, metric_inv, dGdtheta):
    dH0 = grad_logpdf(theta)
    dH1 = -0.5 * np.trace(metric_inv(theta) @ dGdtheta(theta), axis1=1, axis2=2)
    dH2 = []
    for dGdth in dGdtheta(theta):
        dH2.append(0.5 * np.linalg.multi_dot([p.T, metric_inv(theta), dGdth, metric_inv(theta), p]))
    dH2 = np.asarray(dH2)
    dH = dH0 + dH1 + dH2
    return -dH


def leapfrog_step(theta, p, grad_logpdf, metric_inv, dGdtheta, step_size, verbose=False):
    p_e2 = scipy.optimize.fixed_point(
        lambda p_e2: p - step_size / 2 * dHdtheta(theta, p_e2, grad_logpdf, metric_inv, dGdtheta), p,
        xtol=1e-6, maxiter=500, method='del2')
    if verbose:
        print("Momentum at epsilon/2: {}".format(p_e2))
    theta_e = scipy.optimize.fixed_point(
        lambda theta_e: theta + step_size / 2 * (dHdp(theta, p_e2, metric_inv) +
                                                 dHdtheta(theta_e, p_e2, grad_logpdf, metric_inv, dGdtheta)), theta,
        xtol=1e-6, maxiter=500, method='del2')
    if verbose:
        print("Theta at epsilon: {}".format(theta_e))
    p_e = p_e2 - step_size / 2 * dHdtheta(theta_e, p_e2, grad_logpdf, metric_inv, dGdtheta)
    return theta_e, p_e


def leapfrog(theta, p, n_steps, grad_logpdf, metric_inv, dGdtheta, step_size, **kwargs):
    data = [[theta, p]]
    for _ in range(n_steps):
        theta, p = leapfrog_step(theta, p, grad_logpdf, metric_inv, dGdtheta, step_size, **kwargs)
        data.append([theta, p])
    return data[-1]


def momentum(theta, metric):
    return np.random.multivariate_normal(np.zeros_like(theta), metric(theta))


def accept_percentage(theta0, p0, theta, p, logpdf, metric, metric_inv):
    return min(0, -H(theta, p, logpdf, metric, metric_inv) + H(theta0, p0, logpdf, metric, metric_inv))


def mcmc(theta0, momentum, logpdf, grad_logpdf, metric, metric_inv, dGdtheta,
         n_samples=4, n_steps=10, step_size=0.1, **kwargs):
    theta = [np.asarray(theta0)]
    p = [momentum(theta0, metric)]
    for _ in range(n_samples):
        _theta, _p = leapfrog(theta[-1], p[-1], n_steps, grad_logpdf, metric_inv, dGdtheta, step_size, **kwargs)
        alpha = accept_percentage(theta[-1], p[-1], _theta, _p, logpdf, metric, metric_inv)
        if (alpha > 0) or np.log(np.random.rand()) < alpha:
            theta.append(_theta)
            p.append(_p)
        else:
            theta.append(theta[-1])
            p.append(p[-1])
    return theta


if __name__ == '__main__':
    if 0:
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

    if 1:
        dim = 4
        cov = np.eye(dim)
        b = Banana(mean=np.zeros(dim), cov=cov, warp=2, sym=True)
        theta = np.array([0.05417055, 0.02961727, 0.02961727, 0.02961727])
        p = np.array([0.57334575, 0.78403374, 0.78403374, 0.78403374])

        start = time.clock()
        print(leapfrog_step(theta, p, b.grad_logpdf, b.hessian_inv, b.dGdtheta, 0.01))
        print("Time: {}".format(time.clock() - start))

        def neg_hess(hessian):
            def out(*args, **kwargs):
                return -hessian(*args, **kwargs)
            return out

        samples = mcmc(theta, momentum, b.logpdf, b.grad_logpdf, b.G, b.G_inv, b.dGdtheta)
        print(samples)
        print('okay')
