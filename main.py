import numpy as np
import joblib
from skimage import io

import predict
import train

model1 = None
model2 = None
INPUT_PATH = 'input/soccer'
OUTPUT_PATH = 'output/'
MIN_G_NUM = 2
MAX_G_NUM = 10
G_NUM_STEP = 2


# convert to 1D array
def reshape_image(img):
    shape = img.shape
    reshape = img.reshape((-1, 3))

    return shape, reshape


def load_model(suffix):
    global model1, model2
    try:
        model1 = joblib.load(f"{OUTPUT_PATH}models/model1_{str(suffix)}")
        model2 = joblib.load(f"{OUTPUT_PATH}models/model2_{str(suffix)}")
    except FileNotFoundError:
        train.train_model1(suffix, img1)
        train.train_model2(suffix, img1, img2)
        model1 = joblib.load(f"{OUTPUT_PATH}models/model1_{str(suffix)}")
        model2 = joblib.load(f"{OUTPUT_PATH}models/model2_{str(suffix)}")


if __name__ == '__main__':
    for g_num in np.arange(MIN_G_NUM, MAX_G_NUM, step=G_NUM_STEP):
        print(f">>> Number of gaussians: {g_num}")
        load_model(g_num)

        img1_shape, img1 = reshape_image(io.imread(f"{INPUT_PATH}1.jpg"))
        img2_shape, img2 = reshape_image(io.imread(f"{INPUT_PATH}2.jpg"))

        p11 = predict.predict_image(model1, img1, img1_shape[:2], 'M1', 'soccer1', g_num)
        p21 = predict.predict_image(model2, img1, img1_shape[:2], 'M2', 'soccer1', g_num)
        p12 = predict.predict_image(model1, img2, img2_shape[:2], 'M1', 'soccer2', g_num)
        p22 = predict.predict_image(model2, img2, img2_shape[:2], 'M2', 'soccer2', g_num)

        _, truth1 = reshape_image(io.imread(f'{INPUT_PATH}1_mask.png'))
        _, truth2 = reshape_image(io.imread(f'{INPUT_PATH}2_mask.png'))

        predict.compare_result(p11, truth1, 'M1', 'soccer1')
        predict.compare_result(p21, truth1, 'M2', 'soccer1')
        predict.compare_result(p12, truth2, 'M1', 'soccer2')
        predict.compare_result(p22, truth2, 'M2', 'soccer2')
