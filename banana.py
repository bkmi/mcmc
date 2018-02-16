import numpy as np
import scipy.stats


class Banana:
    def __init__(self, mean, cov, warp=1.0):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.warp = warp

        self.det_cov = np.linalg.det(self.cov)
        self.cov_inv = np.linalg.inv(self.cov)
        self.dim = self.mean.size

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

    def hessian(self, x):


        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats
    import seaborn as sns
    import pandas as pd

    dim = 2
    cov = np.eye(dim)
    b = Banana(mean=np.zeros(dim), cov=cov, warp=2)
    data = b.rvs(1000)
    df = pd.DataFrame(data)

    print(b.logpdf(np.ones(2)), b.grad_logpdf(np.ones(2)))
