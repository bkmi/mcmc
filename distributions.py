import numpy as np
import scipy.stats


# Bimodal Normal Distribution in 2D
class bimodal_dist:
    """Returns a frozen bimodal gaussian distribution with equal weighting. Standard scipy stats methods."""

    def __init__(self, mean1, mean2, cov1, cov2):
        self.target_dist1 = scipy.stats.multivariate_normal(mean1, cov1)
        self.target_dist2 = scipy.stats.multivariate_normal(mean2, cov2)
        self.mean1, self.mean2 = mean1, mean2
        self.cov1, self.cov2 = np.asarray(cov1), np.asarray(cov2)
        self.weight = 0.5

    @staticmethod
    def bimodal_gradient_gaussian_2d(x, y, sigx1, sigy1, rho1, mux1, muy1, sigx2, sigy2, rho2, mux2, muy2, weight):
        # from sympy import *
        # x, y = symbols('x y')
        # sigx1, sigy1 = symbols('sigx1 sigy1')
        # rho1, mux1, muy1 = symbols('rho1 mux1 muy1')
        # sigx2, sigy2 = symbols('sigx2 sigy2')
        # rho2, mux2, muy2 = symbols('rho2 mux2 muy2')
        # weight = symbols('weight')
        # bimodal_gaussian1 = 1 / (2 * pi * sigx1 * sigy1 * sqrt(1 - rho1 ** 2)) * exp(-1 / (2 * (1 - rho1) ** 2) * (
        #         (x - mux1) ** 2 / sigx1 ** 2 + (y - muy1) ** 2 / sigy1 ** 2 - 2 * rho1 * (x - mux1) * (y - muy1) / (sigx1 * sigy1)))
        # bimodal_gaussian2 = 1 / (2 * pi * sigx2 * sigy2 * sqrt(1 - rho2 ** 2)) * exp(-1 / (2 * (1 - rho2) ** 2) * (
        #         (x - mux2) ** 2 / sigx2 ** 2 + (y - muy2) ** 2 / sigy2 ** 2 - 2 * rho2 * (x - mux2) * (y - muy2) / (sigx2 * sigy2)))
        # bimodal_gaussian = weight*bimodal_gaussian1 + (1-weight)*bimodal_gaussian2
        # diff(bimodal_gaussian, x)
        # diff(bimodal_gaussian, y)
        dx = -(-weight + 1) * (-2 * rho2 * (-muy2 + y) / (sigx2 * sigy2) + (-2 * mux2 + 2 * x) / sigx2 ** 2) * np.exp(-(
                -2 * rho2 * (-mux2 + x) * (-muy2 + y) / (sigx2 * sigy2) + (-muy2 + y) ** 2 / sigy2 ** 2 + (
                -mux2 + x) ** 2 / sigx2 ** 2) / (2 * (-rho2 + 1) ** 2)) / (
                     4 * np.pi * sigx2 * sigy2 * (-rho2 + 1) ** 2 * np.sqrt(-rho2 ** 2 + 1)) - weight * (
                     -2 * rho1 * (-muy1 + y) / (sigx1 * sigy1) + (-2 * mux1 + 2 * x) / sigx1 ** 2) * np.exp(-(
                -2 * rho1 * (-mux1 + x) * (-muy1 + y) / (sigx1 * sigy1) + (-muy1 + y) ** 2 / sigy1 ** 2 + (
                -mux1 + x) ** 2 / sigx1 ** 2) / (2 * (-rho1 + 1) ** 2)) / (
                     4 * np.pi * sigx1 * sigy1 * (-rho1 + 1) ** 2 * np.sqrt(-rho1 ** 2 + 1))
        dy = -(-weight + 1) * (-2 * rho2 * (-mux2 + x) / (sigx2 * sigy2) + (-2 * muy2 + 2 * y) / sigy2 ** 2) * np.exp(-(
                -2 * rho2 * (-mux2 + x) * (-muy2 + y) / (sigx2 * sigy2) + (-muy2 + y) ** 2 / sigy2 ** 2 + (
                -mux2 + x) ** 2 / sigx2 ** 2) / (2 * (-rho2 + 1) ** 2)) / (
                     4 * np.pi * sigx2 * sigy2 * (-rho2 + 1) ** 2 * np.sqrt(-rho2 ** 2 + 1)) - weight * (
                     -2 * rho1 * (-mux1 + x) / (sigx1 * sigy1) + (-2 * muy1 + 2 * y) / sigy1 ** 2) * np.exp(-(
                -2 * rho1 * (-mux1 + x) * (-muy1 + y) / (sigx1 * sigy1) + (-muy1 + y) ** 2 / sigy1 ** 2 + (
                -mux1 + x) ** 2 / sigx1 ** 2) / (2 * (-rho1 + 1) ** 2)) / (
                     4 * np.pi * sigx1 * sigy1 * (-rho1 + 1) ** 2 * np.sqrt(-rho1 ** 2 + 1))
        return np.asarray([dx, dy])

    @staticmethod
    def bimodal_gradient_log_gaussian_2d(x, y, sigx1, sigy1, rho1, mux1, muy1, sigx2, sigy2, rho2, mux2, muy2, weight):
        # from sympy import *
        # x, y = symbols('x y')
        # sigx1, sigy1 = symbols('sigx1 sigy1')
        # rho1, mux1, muy1 = symbols('rho1 mux1 muy1')
        # sigx2, sigy2 = symbols('sigx2 sigy2')
        # rho2, mux2, muy2 = symbols('rho2 mux2 muy2')
        # weight = symbols('weight')
        # bimodal_gaussian1 = 1 / (2 * pi * sigx1 * sigy1 * sqrt(1 - rho1 ** 2)) * exp(-1 / (2 * (1 - rho1) ** 2) * (
        #         (x - mux1) ** 2 / sigx1 ** 2 + (y - muy1) ** 2 / sigy1 ** 2 - 2 * rho1 * (x - mux1) * (y - muy1) / (sigx1 * sigy1)))
        # bimodal_gaussian2 = 1 / (2 * pi * sigx2 * sigy2 * sqrt(1 - rho2 ** 2)) * exp(-1 / (2 * (1 - rho2) ** 2) * (
        #         (x - mux2) ** 2 / sigx2 ** 2 + (y - muy2) ** 2 / sigy2 ** 2 - 2 * rho2 * (x - mux2) * (y - muy2) / (sigx2 * sigy2)))
        # log_bimodal_gaussian = log(weight*bimodal_gaussian1 + (1-weight)*bimodal_gaussian2)
        # diff(log_bimodal_gaussian, x)
        # diff(log_bimodal_gaussian, y)
        dx = (-(-weight + 1) * (-2 * rho2 * (-muy2 + y) / (sigx2 * sigy2) + (-2 * mux2 + 2 * x) / sigx2 ** 2) * np.exp(
            -(-2 * rho2 * (-mux2 + x) * (-muy2 + y) / (sigx2 * sigy2) + (-muy2 + y) ** 2 / sigy2 ** 2 + (
                    -mux2 + x) ** 2 / sigx2 ** 2) / (2 * (-rho2 + 1) ** 2)) / (
                      4 * np.pi * sigx2 * sigy2 * (-rho2 + 1) ** 2 * np.sqrt(-rho2 ** 2 + 1)) - weight * (
                      -2 * rho1 * (-muy1 + y) / (sigx1 * sigy1) + (-2 * mux1 + 2 * x) / sigx1 ** 2) * np.exp(-(
                -2 * rho1 * (-mux1 + x) * (-muy1 + y) / (sigx1 * sigy1) + (-muy1 + y) ** 2 / sigy1 ** 2 + (
                -mux1 + x) ** 2 / sigx1 ** 2) / (2 * (-rho1 + 1) ** 2)) / (
                      4 * np.pi * sigx1 * sigy1 * (-rho1 + 1) ** 2 * np.sqrt(-rho1 ** 2 + 1))) / (
                     (-weight + 1) * np.exp(-(
                     -2 * rho2 * (-mux2 + x) * (-muy2 + y) / (sigx2 * sigy2) + (-muy2 + y) ** 2 / sigy2 ** 2 + (
                     -mux2 + x) ** 2 / sigx2 ** 2) / (2 * (-rho2 + 1) ** 2)) / (
                             2 * np.pi * sigx2 * sigy2 * np.sqrt(-rho2 ** 2 + 1)) + weight * np.exp(-(
                     -2 * rho1 * (-mux1 + x) * (-muy1 + y) / (sigx1 * sigy1) + (-muy1 + y) ** 2 / sigy1 ** 2 + (
                     -mux1 + x) ** 2 / sigx1 ** 2) / (2 * (-rho1 + 1) ** 2)) / (
                             2 * np.pi * sigx1 * sigy1 * np.sqrt(-rho1 ** 2 + 1)))
        dy = (-(-weight + 1) * (-2 * rho2 * (-mux2 + x) / (sigx2 * sigy2) + (-2 * muy2 + 2 * y) / sigy2 ** 2) * np.exp(
            -(-2 * rho2 * (-mux2 + x) * (-muy2 + y) / (sigx2 * sigy2) + (-muy2 + y) ** 2 / sigy2 ** 2 + (
                    -mux2 + x) ** 2 / sigx2 ** 2) / (2 * (-rho2 + 1) ** 2)) / (
                      4 * np.pi * sigx2 * sigy2 * (-rho2 + 1) ** 2 * np.sqrt(-rho2 ** 2 + 1)) - weight * (
                      -2 * rho1 * (-mux1 + x) / (sigx1 * sigy1) + (-2 * muy1 + 2 * y) / sigy1 ** 2) * np.exp(-(
                -2 * rho1 * (-mux1 + x) * (-muy1 + y) / (sigx1 * sigy1) + (-muy1 + y) ** 2 / sigy1 ** 2 + (
                -mux1 + x) ** 2 / sigx1 ** 2) / (2 * (-rho1 + 1) ** 2)) / (
                      4 * np.pi * sigx1 * sigy1 * (-rho1 + 1) ** 2 * np.sqrt(-rho1 ** 2 + 1))) / (
                     (-weight + 1) * np.exp(-(
                     -2 * rho2 * (-mux2 + x) * (-muy2 + y) / (sigx2 * sigy2) + (-muy2 + y) ** 2 / sigy2 ** 2 + (
                     -mux2 + x) ** 2 / sigx2 ** 2) / (2 * (-rho2 + 1) ** 2)) / (
                             2 * np.pi * sigx2 * sigy2 * np.sqrt(-rho2 ** 2 + 1)) + weight * np.exp(-(
                     -2 * rho1 * (-mux1 + x) * (-muy1 + y) / (sigx1 * sigy1) + (-muy1 + y) ** 2 / sigy1 ** 2 + (
                     -mux1 + x) ** 2 / sigx1 ** 2) / (2 * (-rho1 + 1) ** 2)) / (
                             2 * np.pi * sigx1 * sigy1 * np.sqrt(-rho1 ** 2 + 1)))
        return np.asarray([dx, dy])

    @property
    def cov(self):
        return (self.target_dist1.cov, self.target_dist2.cov)

    @property
    def mean(self):
        return (self.target_dist1.mean, self.target_dist2.mean)

    def rvs(self, samples):
        data1 = self.target_dist1.rvs(round(self.weight * samples))
        data2 = self.target_dist2.rvs(round(abs(1 - self.weight) * samples))
        return np.concatenate((data1, data2), axis=0)

    def pdf(self, state):
        return self.weight * self.target_dist1.pdf(state) + (1 - self.weight) * self.target_dist2.pdf(state)

    def grad_pdf(self, state):
        state = np.asarray(state)
        sigx1, sigy1 = np.sqrt(self.cov1[0, 0]), np.sqrt(self.cov1[1, 1])
        rho1 = self.cov1[0, 1] / (sigx1 * sigy1)
        mux1, muy1 = self.mean1[0], self.mean1[1]
        sigx2, sigy2 = np.sqrt(self.cov2[0, 0]), np.sqrt(self.cov2[1, 1])
        rho2 = self.cov2[0, 1] / (sigx2 * sigy2)
        mux2, muy2 = self.mean2[0], self.mean2[1]
        return bimodal_dist.bimodal_gradient_gaussian_2d(state[..., 0], state[..., 1],
                                                         sigx1, sigy1, rho1, mux1, muy1, sigx2, sigy2, rho2, mux2,
                                                         muy2, self.weight)

    def logpdf(self, state):
        return np.log(self.weight * self.target_dist1.pdf(state) + (1 - self.weight) * self.target_dist2.pdf(state))

    def grad_logpdf(self, state):
        state = np.asarray(state)
        sigx1, sigy1 = np.sqrt(self.cov1[0, 0]), np.sqrt(self.cov1[1, 1])
        rho1 = self.cov1[0, 1] / (sigx1 * sigy1)
        mux1, muy1 = self.mean1[0], self.mean1[1]
        sigx2, sigy2 = np.sqrt(self.cov2[0, 0]), np.sqrt(self.cov2[1, 1])
        rho2 = self.cov2[0, 1] / (sigx2 * sigy2)
        mux2, muy2 = self.mean2[0], self.mean2[1]
        return bimodal_dist.bimodal_gradient_log_gaussian_2d(state[..., 0], state[..., 1],
                                                             sigx1, sigy1, rho1, mux1, muy1, sigx2, sigy2, rho2, mux2,
                                                             muy2, self.weight)


class parabolic_gaussian:
    """Returns a frozen parabolic gaussian distribution, warped along the y axis. Standard scipy stats methods."""

    def __init__(self, mean, cov, warp=0.5):
        self.target_dist = scipy.stats.multivariate_normal(mean, cov)
        self.mean = mean
        self.cov = np.asarray(cov)
        self.warp = warp

    def rvs(self, num_samples):
        samples = self.target_dist.rvs(num_samples)
        samples[..., 0] = samples[..., 0] - self.warp * samples[..., 1] ** 2
        return samples

    @staticmethod
    def warpgaussian(x, y, sigx, sigy, rho, mux, muy, warp):
        warpg = 1 / (2 * np.pi * sigx * sigy * np.sqrt(1 - rho ** 2)) * np.exp(-1 / (2 * (1 - rho) ** 2) * (
                ((x + warp * y ** 2) - mux) ** 2 / sigx ** 2 + (y - muy) ** 2 / sigy ** 2 - 2 * rho * (
                (x + warp * y ** 2) - mux) * (y - muy) / (sigx * sigy)))
        return warpg

    def pdf(self, state):
        # state = np.asarray(state)
        # state[..., 0] = state[..., 0] + self.warp * state[..., 1] ** 2
        # return self.target_dist.pdf(state)

        state = np.asarray(state)
        sigx, sigy = np.sqrt(self.cov[0, 0]), np.sqrt(self.cov[1, 1])
        rho = self.cov[0, 1] / (sigx * sigy)
        mux, muy = self.mean[0], self.mean[1]
        warp = self.warp

        return parabolic_gaussian.warpgaussian(state[..., 0], state[..., 1], sigx, sigy, rho, mux, muy, warp)

    @staticmethod
    def warploggaussian(x, y, sigx, sigy, rho, mux, muy, warp):
        logwarpg = np.log(1 / (2 * np.pi * sigx * sigy * np.sqrt(1 - rho ** 2))) + (-1 / (2 * (1 - rho) ** 2) * (
                ((x + warp * y ** 2) - mux) ** 2 / sigx ** 2 + (y - muy) ** 2 / sigy ** 2 - 2 * rho * (
                (x + warp * y ** 2) - mux) * (y - muy) / (sigx * sigy)))
        return logwarpg

    def logpdf(self, state):
        # state = np.asarray(state)
        # state[..., 0] = state[..., 0] + self.warp * state[..., 1] ** 2

        state = np.asarray(state)
        sigx, sigy = np.sqrt(self.cov[0, 0]), np.sqrt(self.cov[1, 1])
        rho = self.cov[0, 1] / (sigx * sigy)
        mux, muy = self.mean[0], self.mean[1]
        warp = self.warp

        return parabolic_gaussian.warploggaussian(state[..., 0], state[..., 1], sigx, sigy, rho, mux, muy, warp)

    @staticmethod
    def parabolic_gaussian_pdf_gradient_2d(x, y, sigx, sigy, rho, mux, muy, warp):
        # from sympy import *
        # x, y = symbols('x y')
        # sigx, sigy = symbols('sigx sigy')
        # rho, mux, muy = symbols('rho mux muy')
        # warp = symbols('warp')
        #
        # warpg = 1 / (2 * pi * sigx * sigy * sqrt(1 - rho ** 2)) * exp(
        #     -1 / (2 * (1 - rho) ** 2) * (
        #             ((x + warp * y ** 2) - mux) ** 2 / sigx ** 2 +
        #             (y - muy) ** 2 / sigy ** 2 -
        #             2 * rho * ((x + warp * y ** 2) - mux) * (y - muy) / (sigx * sigy)))
        #
        # print(diff(warpg, x))
        # print(diff(warpg, y))
        dx = -(-2 * rho * (-muy + y) / (sigx * sigy) + (-2 * mux + 2 * warp * y ** 2 + 2 * x) / sigx ** 2) * np.exp(-(
                -2 * rho * (-muy + y) * (-mux + warp * y ** 2 + x) / (sigx * sigy) + (-muy + y) ** 2 / sigy ** 2 + (
                -mux + warp * y ** 2 + x) ** 2 / sigx ** 2) / (2 * (-rho + 1) ** 2)) / (
                     4 * np.pi * sigx * sigy * (-rho + 1) ** 2 * np.sqrt(-rho ** 2 + 1))
        dy = -(-4 * rho * warp * y * (-muy + y) / (sigx * sigy) - 2 * rho * (-mux + warp * y ** 2 + x) / (
                sigx * sigy) + (-2 * muy + 2 * y) / sigy ** 2 + 4 * warp * y * (
                       -mux + warp * y ** 2 + x) / sigx ** 2) * np.exp(-(
                -2 * rho * (-muy + y) * (-mux + warp * y ** 2 + x) / (sigx * sigy) + (-muy + y) ** 2 / sigy ** 2 + (
                -mux + warp * y ** 2 + x) ** 2 / sigx ** 2) / (2 * (-rho + 1) ** 2)) / (
                     4 * np.pi * sigx * sigy * (-rho + 1) ** 2 * np.sqrt(-rho ** 2 + 1))
        return np.asarray([dx, dy])

    @staticmethod
    def parabolic_gaussian_logpdf_gradient_2d(x, y, sigx, sigy, rho, mux, muy, warp):
        # from sympy import *
        # x, y = symbols('x y')
        # sigx, sigy = symbols('sigx sigy')
        # rho, mux, muy = symbols('rho mux muy')
        # warp = symbols('warp')
        #
        # warpg = 1 / (2 * pi * sigx * sigy * sqrt(1 - rho ** 2)) * exp(
        #     -1 / (2 * (1 - rho) ** 2) * (
        #             ((x + warp * y ** 2) - mux) ** 2 / sigx ** 2 +
        #             (y - muy) ** 2 / sigy ** 2 -
        #             2 * rho * ((x + warp * y ** 2) - mux) * (y - muy) / (sigx * sigy)))
        # warpg = log(warpg)
        #
        # print(diff(warpg, x))
        # print(diff(warpg, y))
        dx = -(-2 * rho * (-muy + y) / (sigx * sigy) + (-2 * mux + 2 * warp * y ** 2 + 2 * x) / sigx ** 2) / (
                2 * (-rho + 1) ** 2)
        dy = -(-4 * rho * warp * y * (-muy + y) / (sigx * sigy) - 2 * rho * (-mux + warp * y ** 2 + x) / (
                sigx * sigy) + (-2 * muy + 2 * y) / sigy ** 2 + 4 * warp * y * (
                       -mux + warp * y ** 2 + x) / sigx ** 2) / (2 * (-rho + 1) ** 2)
        return np.asarray([dx, dy])

    def grad_pdf(self, state):
        state = np.asarray(state)
        sigx, sigy = np.sqrt(self.cov[0, 0]), np.sqrt(self.cov[1, 1])
        rho = self.cov[0, 1] / (sigx * sigy)
        mux, muy = self.mean[0], self.mean[1]
        warp = self.warp
        return parabolic_gaussian.parabolic_gaussian_pdf_gradient_2d(state[..., 0], state[..., 1], sigx, sigy, rho,
                                                                     mux, muy, warp)

    def grad_logpdf(self, state):
        state = np.asarray(state)
        sigx, sigy = np.sqrt(self.cov[0, 0]), np.sqrt(self.cov[1, 1])
        rho = self.cov[0, 1] / (sigx * sigy)
        mux, muy = self.mean[0], self.mean[1]
        warp = self.warp
        return parabolic_gaussian.parabolic_gaussian_logpdf_gradient_2d(state[..., 0], state[..., 1], sigx, sigy,
                                                                        rho, mux, muy, warp)


class stretched_normal:
    def __init__(self, mean, cov):
        self.target_dist = scipy.stats.multivariate_normal(mean, cov)
        self.mean = mean
        self.cov = np.asarray(cov)

    def rvs(self, num_samples):
        samples = self.target_dist.rvs(num_samples)
        return samples

    def pdf(self, state):
        state = np.asarray(state)
        return self.target_dist.pdf(state)

    def logpdf(self, state):
        state = np.asarray(state)
        return self.target_dist.logpdf(state)

    @staticmethod
    def gradient_gaussian_2d(x, y, sigx, sigy, rho, mux, muy):
        # gaussian = 1/(2 * pi * sigx * sigy * sqrt(1-rho**2)) * exp(-1/(2 * (1-rho)**2) * ((x - mux)**2/sigx**2 + (y - muy)**2/sigy**2 - 2 * rho * (x - mux) * (y - muy)/(sigx * sigy)))
        dx = -(-2 * rho * (-muy + y) / (sigx * sigy) + (-2 * mux + 2 * x) / sigx ** 2) * np.exp(-(
                -2 * rho * (-mux + x) * (-muy + y) / (sigx * sigy) + (-muy + y) ** 2 / sigy ** 2 + (
                -mux + x) ** 2 / sigx ** 2) / (2 * (-rho + 1) ** 2)) / (
                     4 * np.pi * sigx * sigy * (-rho + 1) ** 2 * np.sqrt(-rho ** 2 + 1))
        dy = -(-2 * rho * (-mux + x) / (sigx * sigy) + (-2 * muy + 2 * y) / sigy ** 2) * np.exp(-(
                -2 * rho * (-mux + x) * (-muy + y) / (sigx * sigy) + (-muy + y) ** 2 / sigy ** 2 + (
                -mux + x) ** 2 / sigx ** 2) / (2 * (-rho + 1) ** 2)) / (
                     4 * np.pi * sigx * sigy * (-rho + 1) ** 2 * np.sqrt(-rho ** 2 + 1))
        return np.asarray([dx, dy])

    @staticmethod
    def gradient_log_gaussian_2d(x, y, sigx, sigy, rho, mux, muy):
        # from sympy import *
        # x, y = symbols('x y')
        # sigx, sigy = symbols('sigx sigy')
        # rho, mux, muy = symbols('rho mux muy')
        # log_gaussian = log(1/(2 * pi * sigx * sigy * sqrt(1-rho**2)) * exp(-1/(2 * (1-rho)**2) * ((x - mux)**2/sigx**2 + (y - muy)**2/sigy**2 - 2 * rho * (x - mux) * (y - muy)/(sigx * sigy))))
        # diff(log_gaussian, x)
        # diff(log_gaussian, y)
        dx = -(-2 * rho * (-muy + y) / (sigx * sigy) + (-2 * mux + 2 * x) / sigx ** 2) / (2 * (-rho + 1) ** 2)
        dy = -(-2 * rho * (-mux + x) / (sigx * sigy) + (-2 * muy + 2 * y) / sigy ** 2) / (2 * (-rho + 1) ** 2)
        return np.asarray([dx, dy])

    def grad_pdf(self, state):
        state = np.asarray(state)
        sigx, sigy = np.sqrt(self.cov[0, 0]), np.sqrt(self.cov[1, 1])
        rho = self.cov[0, 1] / (sigx * sigy)
        mux, muy = self.mean[0], self.mean[1]
        return stretched_normal.gradient_gaussian_2d(state[..., 0], state[..., 1], sigx, sigy, rho, mux, muy)

    def grad_logpdf(self, state):
        state = np.asarray(state)
        sigx, sigy = np.sqrt(self.cov[0, 0]), np.sqrt(self.cov[1, 1])
        rho = self.cov[0, 1] / (sigx * sigy)
        mux, muy = self.mean[0], self.mean[1]
        return stretched_normal.gradient_log_gaussian_2d(state[..., 0], state[..., 1], sigx, sigy, rho, mux, muy)


def warped_gradient_gaussian_2d(x, y, sigx, sigy, rho, mux, muy, warp):
    # warp_gaussian = 1 / (2 * pi * sigx * sigy * sqrt(1 - rho ** 2)) * exp(
    #     -1 / (2 * (1 - rho) ** 2) * (
    #             ((x + warp * y * 2) - mux) ** 2 / sigx ** 2 + (y - muy) ** 2 / sigy ** 2 - 2 * rho * (
    #             (x + warp * y * 2) - mux) * (y - muy) / (sigx * sigy)))
    dx = -(-2 * rho * (-muy + y) / (sigx * sigy) + (-2 * mux + 4 * warp * y + 2 * x) / sigx ** 2) * np.exp(-(
            -2 * rho * (-muy + y) * (-mux + 2 * warp * y + x) / (sigx * sigy) + (-muy + y) ** 2 / sigy ** 2 + (
            -mux + 2 * warp * y + x) ** 2 / sigx ** 2) / (2 * (-rho + 1) ** 2)) / (
                 4 * np.pi * sigx * sigy * (-rho + 1) ** 2 * np.sqrt(-rho ** 2 + 1))
    dy = -(-4 * rho * warp * (-muy + y) / (sigx * sigy) - 2 * rho * (-mux + 2 * warp * y + x) / (sigx * sigy) + (
            -2 * muy + 2 * y) / sigy ** 2 + 4 * warp * (-mux + 2 * warp * y + x) / sigx ** 2) * np.exp(-(
            -2 * rho * (-muy + y) * (-mux + 2 * warp * y + x) / (sigx * sigy) + (-muy + y) ** 2 / sigy ** 2 + (
            -mux + 2 * warp * y + x) ** 2 / sigx ** 2) / (2 * (-rho + 1) ** 2)) / (
                 4 * np.pi * sigx * sigy * (-rho + 1) ** 2 * np.sqrt(-rho ** 2 + 1))
    return np.asarray(dx, dy)


def warped_gradient_log_gaussian_2d(x, y, sigx, sigy, rho, mux, muy, warp):
    # from sympy import *
    # x, y = symbols('x y')
    # sigx, sigy = symbols('sigx sigy')
    # rho, mux, muy = symbols('rho mux muy')
    # warp = symbols('warp')
    # warp_log_gaussian = log(1 / (2 * pi * sigx * sigy * sqrt(1 - rho ** 2)) * exp(
    #     -1 / (2 * (1 - rho) ** 2) * (
    #             ((x + warp * y * 2) - mux) ** 2 / sigx ** 2 + (y - muy) ** 2 / sigy ** 2 - 2 * rho * (
    #             (x + warp * y * 2) - mux) * (y - muy) / (sigx * sigy))))
    # diff(warp_log_gaussian, x)
    # diff(warp_log_gaussian, y)
    dx = -(-2 * rho * (-muy + y) / (sigx * sigy) + (-2 * mux + 4 * warp * y + 2 * x) / sigx ** 2) / (
            2 * (-rho + 1) ** 2)
    dy = -(-4 * rho * warp * (-muy + y) / (sigx * sigy) - 2 * rho * (-mux + 2 * warp * y + x) / (sigx * sigy) + (
            -2 * muy + 2 * y) / sigy ** 2 + 4 * warp * (-mux + 2 * warp * y + x) / sigx ** 2) / (2 * (-rho + 1) ** 2)
    return np.asarray(dx, dy)


def warped_normal_2d(mean, covar, samples):
    """Samples from a parabolic warping of a normal distribution in two dimensions.
    Warping:
    (x,y) --> (x+0.05y^2, y)"""
    data = np.random.multivariate_normal(mean, covar, samples)

    def warp(a):
        return np.asarray([a[0] + 0.05 * a[1] ** 2, a[1]])

    data = np.apply_along_axis(warp, 1, data)

    return data


def bimodal_normal_2d(means, covars, weight, samples):
    """Samples from a bimodal normal distribution in two dimensions."""
    data1 = np.random.multivariate_normal(means[0], covars[0], round(weight * samples))
    data2 = np.random.multivariate_normal(means[1], covars[1], round(abs(1 - weight) * samples))
    return np.concatenate((data1, data2), axis=0)


if __name__ == '__main__':
    if 0:
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]
        out = warped_normal_2d(mean, cov, 100)
        print(out, out.shape)

    if 0:
        mean1 = [2, 0]
        cov1 = [[1, 0], [0, 1]]

        mean2 = [-2, 0]
        cov2 = [[1, 0], [0, 1]]

        rv = bimodal_dist(mean1, mean2, cov1, cov2)
        a = np.asarray([[0, 0], [1, 1]])
        print(rv.grad_pdf(a))

    if 1:
        mean = [0, 0]
        cov = [[1, 0], [0, 2]]

        rv = parabolic_gaussian(mean, cov)
        print(rv.rvs(10))
        print(rv.pdf([1, 1]))
        print(rv.logpdf([1, 1]))
