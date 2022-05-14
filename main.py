import numpy as np
import joblib
from skimage import io
import time

from Train import TrainGMM
from Predict import PredictWithGMM
import utils

model1 = None
model2 = None
MIN_G_NUM = 2
MAX_G_NUM = 10
G_NUM_STEP = 2


def load_model(suffix, img1, img2):
    global model1, model2
    try:
        model1 = joblib.load(utils.get_model_path('1', suffix))
        model2 = joblib.load(utils.get_model_path('2', suffix))
    except FileNotFoundError:
        train = TrainGMM(output_path='output/models/')
        train.train_model1(suffix, img1)
        train.train_model2(suffix, img1, img2)
        model1 = joblib.load(utils.get_model_path('1', suffix))
        model2 = joblib.load(utils.get_model_path('2', suffix))


if __name__ == '__main__':
    for g_num in np.arange(MIN_G_NUM, MAX_G_NUM, step=G_NUM_STEP):
        img1_name = 'input/image1.jpg'
        img2_name = 'input/image2.jpg'

        img1_shape, img1 = utils.reshape_image(io.imread(img1_name))
        img2_shape, img2 = utils.reshape_image(io.imread(img2_name))

        load_model(g_num, img1, img2)

        truth1 = utils.convert_to_1d_arr(io.imread('input/image1_mask.png'))
        truth2 = utils.convert_to_1d_arr(io.imread('input/image2_mask.png'))

        print(f"\n=== Image: {img1_name}, Model: M1, Mixture#: {g_num} ===")
        m1_i1_pred = PredictWithGMM(output_path='output/segments/', model_name='M1', img_name='image1', g_num=g_num)
        train_time = time.time()
        p11 = m1_i1_pred.predict_image(model1, img1, img1_shape[:2])
        m1_i1_pred.compare_result(p11, truth1)
        print(f'>>> Prediction time: {time.time() - train_time}')

        print(f"\n=== Image: {img1_name}, Model: M2, Mixture#: {g_num} ===")
        m2_i1_pred = PredictWithGMM(output_path='output/segments/', model_name='M2', img_name='image1', g_num=g_num)
        p21 = m2_i1_pred.predict_image(model2, img1, img1_shape[:2])
        m2_i1_pred.compare_result(p21, truth1)
        print(f'>>> Prediction time: {time.time() - train_time}')

        print(f"\n=== Image: {img2_name}, Model: M1, Mixture#: {g_num} ===")
        m1_i2_pred = PredictWithGMM(output_path='output/segments/', model_name='M1', img_name='image2', g_num=g_num)
        p12 = m1_i2_pred.predict_image(model1, img2, img2_shape[:2])
        m1_i2_pred.compare_result(p12, truth2)
        print(f'>>> Prediction time: {time.time() - train_time}')

        print(f"\n=== Image: {img2_name}, Model: M2, Mixture#: {g_num} ===")
        m2_i2_pred = PredictWithGMM(output_path='output/segments/', model_name='M1', img_name='image2', g_num=g_num)
        p22 = m2_i2_pred.predict_image(model2, img2, img2_shape[:2])
        m2_i2_pred.compare_result(p22, truth2)
        print(f'>>> Prediction time: {time.time() - train_time}')
