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


def compare_result(prediction, ground_truth, model_name, g_num, img_name):
    gmm_is_gaussian = get_playf_gaussian_labels(prediction, ground_truth)
    print('gmm_is_gaussian', end=': ')
    print(gmm_is_gaussian)
    binary_prediction = convert_pred_to_binary(prediction, gmm_is_gaussian)
    correct_num = 0

    for i in range(len(ground_truth)):
        if ground_truth[i] == 255 and binary_prediction[i] == 1:
            correct_num += 1
        # both 0
        elif ground_truth[i] == binary_prediction[i]:
            correct_num += 1

    print(f'Number of correct pixel {correct_num}, total pixel {len(ground_truth)}')

    accuracy = correct_num / len(ground_truth)
    print(f'* Accuracy of {img_name} with {model_name} ({g_num} Gaussian): {accuracy}')
