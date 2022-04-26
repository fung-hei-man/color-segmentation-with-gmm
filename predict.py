import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = 'output/segments/'


def predict_image(model, image, image_shape, model_name, img_name, g_num):
    result = model.predict(image)
    print(f"=== Predict {img_name} with {model_name} ({g_num} Gaussians) ===")
    result_img = result.reshape(image_shape)

    plt.title(f"Predict {img_name} with {model_name} ({g_num} Gaussians)")
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
    one_d_gt = ground_truth[:, 0]

    for i in range(len(one_d_gt)):
        if one_d_gt[i] == 255:
            playf_bins[prediction[i]] += 1

    print('pred_bins')
    print(pred_bins)
    print('playf_bins')
    print(playf_bins)

    gmm_is_gaussion = [playf_bins[i]/pred_bins[i] > 0.5 for i in range(len(playf_bins))]
    gmm_index = np.array(range(len(pred_bins)))

    return gmm_index[gmm_is_gaussion]


def compare_result(prediction, ground_truth, model_name, img_name):
    gmm_is_gaussion = get_playf_gaussian_labels(prediction, ground_truth)
    print('gmm_is_gaussion')
    print(gmm_is_gaussion)
