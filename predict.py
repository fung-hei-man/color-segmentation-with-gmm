import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = 'output/segments/'


def predict_image(model, image, image_shape, model_name, img_name, g_num):
    result = model.predict(image)
    result_img = result.reshape(image_shape)

    plt.title(f"Color Segment of {img_name} with {model_name} ({g_num} Gaussians)")
    plt.imshow(result_img), plt.axis('off')

    plt.savefig(f"{OUTPUT_PATH}{img_name}_{model_name}_{g_num}.png".lower())
    plt.show()
    plt.clf()

    return result


def compare_result(prediction, ground_truth, model_name, g_num, img_name):
    gmm_is_gaussian = get_playf_gaussian_labels(prediction, ground_truth)
    print('gmm_is_gaussian', end=': ')
    print(gmm_is_gaussian)
    binary_prediction = convert_pred_to_binary(prediction, gmm_is_gaussian)
    t_pos = 0
    t_neg = 0
    f_pos = 0
    f_neg = 0

    for i in range(len(ground_truth)):
        if ground_truth[i] == 255 and binary_prediction[i] == 1:
            t_pos += 1
        elif ground_truth[i] == 255 and binary_prediction[i] == 0:
            f_neg += 1
        elif ground_truth[i] == 0 and binary_prediction[i] == 1:
            f_pos += 1
        elif ground_truth[i] == 0 and binary_prediction[i] == 0:
            t_neg += 1

    calculate_performance(len(ground_truth), t_pos, t_neg, f_pos, f_neg, img_name, model_name, g_num)


# white (255) = playing field in mask
def get_playf_gaussian_labels(prediction, ground_truth):
    pred_bins = np.bincount(prediction)
    # len(pred_bins) = g_nums
    playf_bins = np.zeros(len(pred_bins))

    for i in range(len(ground_truth)):
        if ground_truth[i] == 255:
            playf_bins[prediction[i]] += 1

    print('pred_bins', end=': ')
    print(pred_bins)
    print('playf_bins', end=': ')
    print(playf_bins)

    gmm_is_gaussian = [playf_bins[i]/pred_bins[i] > 0.7 if playf_bins[i] != 0 and pred_bins[i] != 0 else False for i in range(len(playf_bins))]
    gmm_index = np.array(range(len(pred_bins)))
    return gmm_index[gmm_is_gaussian]


# convert to 1 if predicted label is in gmm_is_gaussian
def convert_pred_to_binary(prediction, gmm_is_gaussian):
    for i in range(len(prediction)):
        if prediction[i] in gmm_is_gaussian:
            prediction[i] = 1
        else:
            prediction[i] = 0

    return prediction


def calculate_performance(total, t_pos, t_neg, f_pos, f_neg, img_name, model_name, g_num):
    print(f'* Result of {img_name} with {model_name} ({g_num} Gaussian) *')
    print(f'  Accuracy: {(t_pos + t_neg) / total} ')
    print(f'  Precision: {t_pos / (t_pos + f_pos)} ')
    print(f'  Recall: {t_pos / (t_pos + f_neg)} ')
