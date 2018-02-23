import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import sympy as sm
import time

import autograd.numpy as npa
import autograd as ag

from banana import Banana, autogradBanana


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


def leapfrog_step(theta, p, grad_logpdf, metric_inv, dGdtheta, step_size, maxiter=10):
    try:
        p_e2 = scipy.optimize.fixed_point(
            lambda p_e2: p - (step_size / 2) * dHdtheta(theta, p_e2, grad_logpdf, metric_inv, dGdtheta),
            p, xtol=1e-8, maxiter=maxiter, method='del2')
        theta_e = scipy.optimize.fixed_point(
            lambda theta_e: theta + (step_size / 2) * (dHdp(theta, p_e2, metric_inv) + dHdp(theta_e, p_e2, metric_inv)),
            theta, xtol=1e-8, maxiter=maxiter, method='del2')
        p_e = p_e2 - (step_size / 2) * dHdtheta(theta_e, p_e2, grad_logpdf, metric_inv, dGdtheta)
    except RuntimeError:
        return None
    return theta_e, p_e


def leapfrog(theta, p, n_steps, grad_logpdf, metric_inv, dGdtheta, step_size, **kwargs):
    data = [[theta, p]]
    for _ in range(n_steps):
        next = leapfrog_step(theta, p, grad_logpdf, metric_inv, dGdtheta, step_size, **kwargs)
        if next is None:
            return None
        else:
            theta, p = next
        data.append([theta, p])
    return data[-1]


def momentum(theta, metric):
    return np.random.multivariate_normal(np.zeros_like(theta), metric(theta))


def accept_percentage(theta0, p0, theta, p, logpdf, metric, metric_inv):
    return min(0, -H(theta, p, logpdf, metric, metric_inv) + H(theta0, p0, logpdf, metric, metric_inv))


def mcmc(theta0, momentum, logpdf, grad_logpdf, metric, metric_inv, dGdtheta,
         n_samples=20, n_steps=10000, step_size=0.0005, **kwargs):
    theta = [np.asarray(theta0)]
    for _ in range(n_samples):
        p = momentum(theta[-1], metric)
        next = leapfrog(theta[-1], p, n_steps, grad_logpdf, metric_inv, dGdtheta, step_size, **kwargs)
        if next is None:
            print('Problem integrating when \ntheta = {}\np = {}'.format(theta[-1], p))
            continue
        else:
            _theta, _p = next

        alpha = accept_percentage(theta[-1], p, _theta, _p, logpdf, metric, metric_inv)
        if (alpha > 0) or np.log(np.random.rand()) < alpha:
            theta.append(_theta)
        else:
            theta.append(theta[-1])
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

    dim = 4
    cov = np.eye(dim)
    b = Banana(mean=np.zeros(dim), cov=cov, warp=15, sym=False, special=True)
    ab = autogradBanana(mean=npa.zeros(dim), cov=npa.eye(dim), warp=15)
    theta = np.array([5, -3, 2, 4])


    def sample_save(rounds,
                    samples_file='samples_w15_auto.npy',
                    time_file='samples_w15_auto.npy',
                    samples_per_round=10, auto=False, verbose=True):
        for _ in range(rounds):
            samples = np.load(samples_file)

            start = time.clock()
            if not auto:
                more = mcmc(samples[-1], momentum, b.logpdf, b.grad_logpdf, b.G, b.G_inv, b.dGdtheta,
                            n_samples=samples_per_round)
            else:
                samples = mcmc(samples[-1], momentum, ab.logpdf, ab.grad_logpdf, ab.G, ab.G_inv, ab.dGdtheta,
                               n_samples=samples_per_round,
                               n_steps=2000, step_size=0.0003)
            finished = time.clock() - start
            samples = np.append(samples, more, axis=0)
            np.save(samples_file, np.asarray(samples))

            t = np.load(time_file)
            t = np.append(t, np.array(finished))
            np.save(time_file, t)
            if verbose:
                print('\n---round---\n')

    if 1:
        # basic
        samp_file = 'samples_w15_off.npy'
        time_file = 'time_w15_off.npy'

        start = time.clock()

        samples = mcmc(theta, momentum, b.logpdf, b.grad_logpdf, b.G, b.G_inv, b.dGdtheta, n_samples=10)

        print("Time: {}".format(time.clock() - start))
        np.save(time_file, np.array(time.clock() - start))

        print(samples)
        np.save(samp_file, np.asarray(samples))

        sample_save(50,
                    samples_file=samp_file,
                    time_file=time_file)

    if 0:
        # with autograd
        samp_file = 'samples_w15_off.npy'
        time_file = 'time_w15_off.npy'

        start = time.clock()
        samples = mcmc(theta, momentum, ab.logpdf, ab.grad_logpdf, ab.G, ab.G_inv, ab.dGdtheta,
                       n_samples=2,
                       n_steps=1000, step_size=0.0005)
        print("Time: {}".format(time.clock() - start))
        np.save(time_file, np.array(time.clock() - start))

        print(samples)
        np.save(samp_file, np.asarray(samples))

        sample_save(10, auto=True, samples_file=samp_file, time_file=time_file)

