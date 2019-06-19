
import matplotlib.pyplot as plt
import numpy as np

class Distance:

    def l2(x,y):
        diff = x-y
        return diff.dot(diff)

    def l1(x,y):
        return np.sum(np.abs(x-y))

    def k_gaussian(x, y, sigma=1.):
        diff = x - y
        return np.exp(-1.*diff.dot(diff)/(2.*sigma**2))

    def k_pseudo_gaussian(x, y, sigma=1.):
        return np.exp(-1.*x.dot(y)/(2.*sigma**2))

    def get_gaussian_k(sigma=1.):
        return lambda x,y: np.exp(-1.*(x-y).dot(x-y)/(2.*sigma**2))

    def get_pseudo_gaussian_k(sigma=1.):
        return lambda x,y: np.exp(1.*x.dot(y)/(2.*sigma**2))

    def get_polynomial_k(bias=0, exponent=1):
        return lambda x,y: (x.dot(y) + bias)**exponent


    def k_dot(x,y):
        return x.dot(y)


def get_pairwise_distance(data, dist_func):
    sq_l2_norm = np.zeros((len(data), len(data)))
    for idx1, ele1 in enumerate(data):
        for idx2, ele2 in enumerate(data):
            sq_l2_norm[idx1][idx2] = dist_func(ele1,ele2)
    return sq_l2_norm
