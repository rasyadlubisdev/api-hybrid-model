import numpy as np
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt


def sliding_window(image, patch_size, step_size):
    patches = []
    coords = []
    h, w, _ = image.shape
    for y in range(0, h - patch_size[0] + 1, step_size):
        for x in range(0, w - patch_size[1] + 1, step_size):
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            patches.append(patch)
            coords.append((x, y))
    return patches, coords


def predict_with_uncertainty(f_model, images, n_iter=50):
    predictions = np.array([f_model.predict(images) for _ in range(n_iter)])
    mean_prediction = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    return mean_prediction, uncertainty


def process_large_image(image_path, model, patch_size, step_size, class_labels):
    img = image.load_img(image_path, target_size=(512, 512))
    img = image.img_to_array(img) / 255.0

    patches, coords = sliding_window(img, patch_size, step_size)
    patches = np.array(patches)

    mean_preds, uncertainties = predict_with_uncertainty(model, patches, n_iter=50)

    pred_classes = np.argmax(mean_preds, axis=1)
    confidences = np.max(mean_preds, axis=1)

    results = []
    for (x, y), pred_class, confidence, uncertainty in zip(coords, pred_classes, confidences, uncertainties):
        results.append({
            "x": x,
            "y": y,
            "width": patch_size[1],
            "height": patch_size[0],
            "label": class_labels[pred_class],
            "confidence": confidence,
            "uncertainty": uncertainty.max()
        })

    return results