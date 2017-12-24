import numpy as np
import scipy.stats

# Bimodal Normal Distribution in 2D
class bimodal_dist:
    """Returns a frozen bimodal gaussian distribution with equal weighting. Standard scipy stats methods."""
    def __init__(self, mean1, mean2, cov1, cov2):
        self.target_dist1 = scipy.stats.multivariate_normal(mean1, cov1)
        self.target_dist2 = scipy.stats.multivariate_normal(mean2, cov2)
        self.weight = 0.5
        
    @property
    def cov(self):
        return (self.target_dist1.cov, self.target_dist2.cov)
    
    @property
    def mean(self):
        return (self.target_dist.mean, self.target_dist2.mean)
    
    def rvs(self, samples):
        data1 = self.target_dist1.rvs(round(self.weight * samples))
        data2 = self.target_dist2.rvs(round(abs(1 - self.weight) * samples))        
        return np.concatenate((data1, data2), axis=0)
    
    def pdf(self, state):
        return self.weight*(self.target_dist1.pdf(state) + self.target_dist2.pdf(state))
    
    def logpdf(self, state):
        return np.log(self.weight*(self.target_dist1.pdf(state) + self.target_dist2.pdf(state)))


def warped_normal_2d(mean, covar, samples):
    """Samples from a parabolic warping of a normal distribution in two dimensions.
    Warping:
    (x,y) --> (x+0.05y^2, y)"""
    data = np.random.multivariate_normal(mean, covar, samples)

    def warp(a):
        return np.asarray([a[0] + 0.05 * a[1]**2, a[1]])

    data = np.apply_along_axis(warp, 1, data)

    return data


def bimodal_normal_2d(means, covars, weight, samples):
    """Samples from a bimodal normal distribution in two dimensions."""
    data1 = np.random.multivariate_normal(means[0], covars[0], round(weight * samples))
    data2 = np.random.multivariate_normal(means[1], covars[1], round(abs(1 - weight) * samples))
    return np.concatenate((data1, data2), axis=0)


if __name__ == '__main__':
    mean = [0, 0]
    cov = [[1, 0], [0, 100]]
    out = warped_normal_2d(mean, cov, 100)
    print( out, out.shape)
