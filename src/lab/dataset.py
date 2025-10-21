import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def mnist(n=999999):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    datax = digits.images.reshape((n_samples, -1))
    # Normaliza para [-1, 1]
    datax = 2 * (datax - datax.min()) / (datax.max() - datax.min()) - 1
    return datax.astype(np.float32)[:n], digits.target[:n]
