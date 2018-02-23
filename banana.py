import numpy as np
import pandas as pd
import scipy.stats
import sympy as sm

import autograd.numpy as npa
import autograd as ag


class Banana:
    def __init__(self, mean, cov, warp=1.0, sym=False, special=True):
        """special means dim=4, cov=np.eye(4)"""
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.warp = warp
        self.sym = sym
        self.special = special

        self.det_cov = np.linalg.det(self.cov)
        self.cov_inv = np.linalg.inv(self.cov)
        self.dim = self.mean.size
        if self.sym & (self.dim != 4):
            raise NotImplementedError('When using sym, we must work in 4d.')

        self._sym_hess = None

    def rvs(self, num_samples):
        samples = scipy.stats.multivariate_normal(self.mean, self.cov).rvs(num_samples)
        samples = self.transform(samples, add=False)
        return samples

    def transform(self, x, add=True):
        x = np.asarray(x)
        shape_in = x.shape
        x = x.reshape(-1, self.dim)
        y = self.warp * x[:, 0] ** 2
        y = y[:, None] * np.ones_like(x)
        y[:, 0] = 0
        if add:
            return (x + y).reshape(shape_in)
        else:
            return (x - y).reshape(shape_in)

    def logpdf(self, x):
        x = self.transform(x)
        x = x.reshape(-1, self.dim)
        y = (x - self.mean).T
        matrix_calc = 0.5 * np.dot(y.T, np.dot(self.cov_inv, y))
        if matrix_calc.ndim > 1:
            matrix_calc = matrix_calc.diagonal()
        return -(np.log(np.sqrt(self.det_cov * (2 * np.pi) ** self.dim)) + matrix_calc)

    def grad_logpdf(self, x):
        x = self.transform(x)
        y = (x - self.mean).T
        y = np.tile(y, (self.dim, 1)).T
        dy = np.eye(self.dim)
        dy[0, 1:] = 2 * self.warp * x[0]
        dy = dy.T
        grad = -0.5 * (np.dot(dy.T, np.dot(self.cov_inv, y)) + np.dot(y.T, np.dot(self.cov_inv, dy)))
        return grad.diagonal()

    def jacobian_transform(self, x, det=True):
        jac = np.eye(self.dim)
        jac[0, 1:] = 2 * self.warp * x[0, ...]
        if det:
            return np.linalg.det(jac)
        else:
            return jac

    @property
    def _hess(self):
        if self._sym_hess is None:
            x0, x1, x2, x3, w = sm.symbols('x0 x1 x2 x3 w')
            cov = sm.Matrix(self.cov.tolist())
            cov_inv = cov ** -1
            mu = sm.Matrix(self.mean.tolist())
            xx = sm.Matrix([[x0],
                            [x1 + w * x0 ** 2],
                            [x2 + w * x0 ** 2],
                            [x3 + w * x0 ** 2]])
            xx = xx - mu
            logpdf = -(sm.log(sm.sqrt(cov.det() * (2 * np.pi) ** 4)) + (1 / 2) * (xx.T * (cov_inv * xx))[0])
            self._sym_hess = [[logpdf.diff(i).diff(j) for i in (x0, x1, x2, x3)] for j in (x0, x1, x2, x3)]
        return self._sym_hess

    def _special_hess(self, x):
        x0, x1, x2, x3, w = x[0], x[1], x[2], x[3], self.warp
        out = np.array([[-12.0 * w ** 2 * x0 ** 2 - 1.0 * w * (1.0 * w * x0 ** 2 + 1.0 * x1) - 1.0 * w * (
                    w * x0 ** 2 + x1) - 1.0 * w * (1.0 * w * x0 ** 2 + 1.0 * x2) - 1.0 * w * (
                                     w * x0 ** 2 + x2) - 1.0 * w * (1.0 * w * x0 ** 2 + 1.0 * x3) - 1.0 * w * (
                                     w * x0 ** 2 + x3) - 1.0, -2.0 * w * x0, -2.0 * w * x0, -2.0 * w * x0],
                        [-2.0 * w * x0, -1., 0, 0],
                        [-2.0 * w * x0, 0, -1., 0],
                        [-2.0 * w * x0, 0, 0, -1.]])
        return out

    def hessian(self, x, sym=None, special=None):
        if sym is None:
            sym = self.sym
        if special is None:
            special = self.special

        if sym and not special:
            x0, x1, x2, x3, w = sm.symbols('x0 x1 x2 x3 w')
            pt = {x0: x[0], x1: x[1], x2: x[2], x3: x[3], w: self.warp}
            hess = np.asarray([[j.subs(pt) for j in i] for i in self._hess])
            return hess.astype(np.float64)
        elif special:
            return self._special_hess(x)
        else:
            x = self.transform(x)
            y = (x - self.mean).T
            y = np.tile(y, (self.dim, 1)).T

            dy = np.eye(self.dim)
            dy[0, 1:] = 2 * self.warp * x[0]
            # two are necessary for dydj and dydk
            dy = dy.T
            dy2 = dy.T

            ddy = np.zeros((self.dim, self.dim, self.dim))
            ddy[0, 0, 1:] = 2 * self.warp

            hess = -0.5 * (np.dot(ddy.T, np.dot(self.cov_inv, y)) +
                           np.dot(dy.T, np.dot(self.cov_inv, dy2)) +
                           np.dot(dy2.T, np.dot(self.cov_inv, dy)) +
                           np.dot(y.T, np.dot(self.cov_inv, ddy)))
            # You could also do this element-wise in python fairly easily.
            return hess.diagonal()

    def G(self, x, **kwargs):
        return -self.hessian(x, **kwargs)

    def hessian_inv(self, x, **kwargs):
        return np.linalg.inv(self.hessian(x, **kwargs))

    def G_inv(self, x, **kwargs):
        return -self.hessian_inv(x, **kwargs)

    @property
    def _dGdtheta(self):
        x0, x1, x2, x3, w = sm.symbols('x0 x1 x2 x3 w')
        dGdtheta = []
        for x in [x0, x1, x2, x3]:
            dGdtheta.append([[j.diff(x) for j in i] for i in self._hess])
        return dGdtheta

    def _special_dGdtheta(self, x):
        x0, x1, x2, x3, w = x[0], x[1], x[2], x[3], self.warp
        out = np.array([[[-36.0 * w ** 2 * x0, -2.0 * w, -2.0 * w, -2.0 * w],
                         [-2.0 * w, 0, 0, 0],
                         [-2.0 * w, 0, 0, 0],
                         [-2.0 * w, 0, 0, 0]],
                        [[-2.0 * w, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]],
                        [[-2.0 * w, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]],
                        [[-2.0 * w, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]]])
        return out

    def dGdtheta(self, x, sym=None, special=None):
        if sym is None:
            sym = self.sym
        if special is None:
            special = self.special

        if sym and not special:
            x0, x1, x2, x3, w = sm.symbols('x0 x1 x2 x3 w')
            pt = {x0: x[0], x1: x[1], x2: x[2], x3: x[3], w: self.warp}
            dGdtheta = np.asarray([[[j.subs(pt) for j in i] for i in k] for k in self._dGdtheta])
            return -dGdtheta.astype(np.float64)
        elif special:
            return -self._special_dGdtheta(x)
        else:
            # Use G instead of hessian
            raise NotImplementedError()


class autogradBanana:
    def __init__(self, mean, cov, warp=1.0):
        self.mean = npa.asarray(mean)
        self.cov = npa.asarray(cov)
        self.warp = warp

        self._grad_logpdf = ag.grad(self.logpdf)
        self._hessian = ag.hessian(self.logpdf)
        self._dGdtheta = ag.hessian(ag.grad(self.logpdf))

        self.det_cov = npa.linalg.det(self.cov)
        self.cov_inv = npa.linalg.inv(self.cov)
        self.dim = self.mean.size

    def rvs(self, num_samples):
        samples = scipy.stats.multivariate_normal(self.mean, self.cov).rvs(num_samples)
        samples = self.transform(samples, add=False)
        return samples

    def transform(self, x, add=True):
        y = np.array([0,
                      self.warp * x[0] ** 2,
                      self.warp * x[0] ** 2,
                      self.warp * x[0] ** 2])
        # y = self.warp * x[0] ** 2
        # y = y * np.ones_like(x)
        # y[0] = 0
        if add:
            return (x + y)
        else:
            return (x - y)

    def logpdf(self, x):
        # x = self.transform(x)
        u = npa.array([0,
                      self.warp * x[0] ** 2,
                      self.warp * x[0] ** 2,
                      self.warp * x[0] ** 2])
        z = x + u
        y = (z - self.mean)
        matrix_calc = 0.5 * npa.dot(y.T, npa.dot(self.cov_inv, y))
        return -(npa.log(npa.sqrt(self.det_cov * (2 * npa.pi) ** self.dim)) + matrix_calc)

    def grad_logpdf(self, x):
        return self._grad_logpdf(x)

    def hessian(self, x):
        return self._hessian(x)

    def G(self, x):
        return -self.hessian(x)

    def G_inv(self, x):
        return npa.linalg.inv(self.G(x))

    def dGdtheta(self, x):
        return self._dGdtheta(x)




def test_calcs():
    cov = [[1.0, 0.0, 0.5, 0.0],
           [0.0, 1.0, 0.0, 0.2],
           [0.5, 0.0, 0.8, 0.0],
           [0.0, 0.2, 0.0, 1.0]]
    mean = [1, .4, -1.2, -0.8]
    warp = 2.3
    x = [2, 3, 4, 1]

    def sym(x, mean, cov, warp):
        x0, x1, x2, x3, w = sm.symbols('x0 x1 x2 x3 w')
        pt = {x0: x[0], x1: x[1], x2: x[2], x3: x[3], w: warp}
        cov = sm.Matrix(cov)
        cov_inv = cov ** -1
        mu = sm.Matrix(mean)
        xx = sm.Matrix([[x0],
                       [x1 + w * x0 ** 2],
                       [x2 + w * x0 ** 2],
                       [x3 + w * x0 ** 2]])
        xx = xx - mu
        logpdf = -(sm.log(sm.sqrt(cov.det() * (2 * np.pi) ** 4)) + (1/2) * (xx.T * (cov_inv * xx))[0])
        grad = [logpdf.diff(i) for i in (x0, x1, x2, x3)]
        hess = [[logpdf.diff(i).diff(j) for i in (x0, x1, x2, x3)] for j in (x0, x1, x2, x3)]

        logpdf_eval = logpdf.subs(pt)
        grad_eval = [i.subs(pt) for i in grad]
        hess_eval = [[j.subs(pt) for j in i] for i in hess]

        return logpdf_eval, grad_eval, hess_eval

    b = Banana(mean, cov, warp)

    p, g, h = b.logpdf(x), b.grad_logpdf(x), b.hessian(x)
    sp, sg, sh = (np.asarray(i) for i in sym(x, mean, cov, warp))

    print(p, sp)
    print(g, sg)
    print(h, sh)


def main():
    dim = 4
    cov = npa.eye(dim)
    # b = Banana(mean=np.zeros(dim), cov=cov, warp=2, sym=False, special=True)
    # data = b.rvs(1000)
    # print(b.logpdf(np.ones(dim)), b.grad_logpdf(np.ones(dim)))
    # print(b.hessian(np.ones(dim)))

    ab = autogradBanana(mean=npa.zeros(dim), cov=cov, warp=2)
    print(ab.logpdf(npa.ones(dim)))
    print(ab.grad_logpdf(npa.ones(dim)))
    print(ab.hessian(npa.ones(dim)))
    print(ab.G_inv(npa.ones(dim)))
    print(ab.dGdtheta(npa.ones(dim)))
    test_calcs()


if __name__ == '__main__':
    main()
