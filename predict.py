import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = 'output/segments/'


def predict_image(model, image, image_shape, model_name, img_name, g_num):
    result = model.predict(image)
    print(f"===== Predict {img_name} with {model_name} ({g_num} Gaussians) =====")
    print(result)
    result_img = result.reshape(image_shape)

    plt.title(f"Predict {img_name} with {model_name} ({g_num} Gaussians)")
    plt.imshow(result_img), plt.axis('off')

    plt.savefig(f"{OUTPUT_PATH}{img_name}_{model_name}_{g_num}.png".lower())
    plt.show()
    plt.clf()


def convert_ground_truth(ground_truth):
    ground_truth = ground_truth.reshape((-1, 3))
    return ground_truth


def compare_result(label, ground_truth, model_name, img_name):
    ground_truth = convert_ground_truth(ground_truth)
    print(f'=== {img_name} ===')
    print(ground_truth)
