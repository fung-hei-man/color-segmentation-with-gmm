import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
import time

OUTPUT_PATH = 'output/models/'


def train_model1(g_num, img1):
    train_time = time.time()
    m1 = train_gmm(g_num, img1)
    joblib.dump(m1, f"{OUTPUT_PATH}model1_{g_num}")
    print(f'Training time of M1 ({g_num} gaussians): {time.time() - train_time}')


def train_model2(g_num, img1, img2):
    train_time = time.time()
    img = np.concatenate([img1, img2])
    m2 = train_gmm(g_num, img)
    joblib.dump(m2, f"{OUTPUT_PATH}model2_{g_num}")
    print(f'Training time of M2 ({g_num} gaussians): {time.time() - train_time}')


def train_gmm(g_num, train_img):
    return GaussianMixture(n_components=g_num, covariance_type='full').fit(train_img)
