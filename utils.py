# convert to 1D array
def reshape_image(img):
    shape = img.shape
    reshape = img.reshape((-1, 3))

    return shape, reshape


def convert_to_1d_arr(img):
    return img.reshape((-1, 3))[:, 0]


def get_model_path(model, suffix):
    return f'output/models/model{model}_{str(suffix)}'
