import matplotlib.pyplot as plt
import numpy as np


class PredictWithGMM:
    def __init__(self, output_path, model_name, img_name, g_num):
        self.output_path = output_path
        self.model_name = model_name
        self.img_name = img_name
        self.g_num = g_num

    def predict_image(self, model, image, image_shape):
        result = model.predict(image)
        result_img = result.reshape(image_shape)

        plt.title(f"Color Segment of {self.img_name} with {self.model_name} ({self.g_num} Gaussians)")
        plt.imshow(result_img), plt.axis('off')

        plt.savefig(f"{self.output_path}{self.img_name}_{self.model_name}_{self.g_num}.png".lower())
        plt.show()
        plt.clf()

        return result

    def compare_result(self, prediction, ground_truth):
        gmm_is_gaussian = self.get_playf_gaussian_labels(prediction, ground_truth)
        print('gmm_is_gaussian', end=': ')
        print(gmm_is_gaussian)
        binary_prediction = self.convert_pred_to_binary(prediction, gmm_is_gaussian)
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

        self.calculate_performance(len(ground_truth), t_pos, t_neg, f_pos, f_neg)

    # white (255) = playing field in mask
    def get_playf_gaussian_labels(self, prediction, ground_truth):
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
    def convert_pred_to_binary(self, prediction, gmm_is_gaussian):
        for i in range(len(prediction)):
            if prediction[i] in gmm_is_gaussian:
                prediction[i] = 1
            else:
                prediction[i] = 0

        return prediction

    def calculate_performance(self, total, t_pos, t_neg, f_pos, f_neg):
        print(f'>> Result of {self.img_name} with {self.model_name} ({self.g_num} Gaussian) <<')
        print(f'>>> Accuracy: {(t_pos + t_neg) / total} ')
        print(f'>>> Precision: {t_pos / (t_pos + f_pos)} ')
        print(f'>>> Recall: {t_pos / (t_pos + f_neg)} ')
