import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def load_image_phase_magnitude(grayscale_path, magnitudes_path, phases_path, image_number):
    """
    Function to load the image, magnitude and phase from the respective paths with a given image number

    :param grayscale_path: path to the grayscale images
    :param magnitudes_path: path to the magnitudes
    :param phases_path: path to the phases
    :param image_number: number of the image to load
    :return: tuple of the grayscale image, magnitude and phase of the image
    """

    pixels = load_img(os.path.join(grayscale_path, f"image_{image_number}.jpg"), target_size=(128, 128))
    pixels = img_to_array(pixels)
    magnitude = np.load(os.path.join(magnitudes_path, f"image_{image_number}.npy"))
    phase = np.load(os.path.join(phases_path, f"image_{image_number}.npy"))

    return pixels, magnitude, phase


def load_custom_model(model_path):
    """
    Function to load a custom model from the given path

    :param model_path: path to the model
    :return: the loaded model
    """

    cust = {"InstanceNormalization": InstanceNormalization}

    return load_model(model_path, cust)


def prepare_magnitude_or_phase(magnitude_or_phase):
    """
    Function to prepare the magnitude or phase for the model input

    :param magnitude_or_phase: the magnitude or phase to change
    :return: the prepared magnitude or phase
    """

    model_input = magnitude_or_phase[:, :, np.newaxis]
    model_input = np.repeat(model_input, 3, axis=2)
    model_input = model_input[np.newaxis, :, :, :]

    return model_input


def predict(model, model_input):
    """
    Function to predict the output of the model with the given input

    :param model: the model to predict with
    :param model_input: the input to the model
    :return: the predicted output
    """

    return np.squeeze(model.predict(model_input))
