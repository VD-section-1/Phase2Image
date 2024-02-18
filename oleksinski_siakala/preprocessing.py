import cv2
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot
from PIL import Image
from tqdm import tqdm


def get_image_paths(source_path):
    """
    Function to get the paths of the images in the given directory

    :param source_path: path to the directory with the images
    :return: list of the paths of the images in the directory
    """

    files = os.listdir(source_path)
    image_files = [file for file in files if file.lower().endswith(".jpg")]

    return image_files


def rename_images(source_path):
    """
    Function to rename the images in the given directory to image_{i}.jpg

    :param source_path: path to the directory with the images
    """

    image_files = get_image_paths(source_path)

    for i, image_file in enumerate(tqdm(image_files, desc="Renaming images")):
        old_path = os.path.join(source_path, image_file)
        new_name = f"image_{i}.jpg"
        new_path = os.path.join(source_path, new_name)
        os.rename(old_path, new_path)


def convert_to_grayscale(source_path, destination_path):
    """
    Function to convert the images in the given directory to grayscale and save them in the destination directory

    :param source_path: path to the directory with the images
    :param destination_path: path to the directory to save the grayscale images
    """

    image_files = get_image_paths(source_path)

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for image_file in tqdm(image_files, desc="Converting to grayscale"):
        image_path = os.path.join(source_path, image_file)
        output_path = os.path.join(destination_path, image_file)
        image = Image.open(image_path)
        grayscale_image = image.convert("L")
        grayscale_image.save(output_path)


def generate_phases_and_magnitudes(source_path, phases_path, magnitudes_path):
    """
    Function to generate the phases and magnitudes of the images in the given directory and save them in the
    respective directories

    :param source_path: path to the directory with the images
    :param phases_path: path to the directory to save the phases
    :param magnitudes_path: path to the directory to save the magnitudes
    """

    image_files = get_image_paths(source_path)

    if not os.path.exists(phases_path):
        os.makedirs(phases_path)

    if not os.path.exists(magnitudes_path):
        os.makedirs(magnitudes_path)

    for image_file in tqdm(image_files, desc="Generating phases and magnitudes"):
        image_path = os.path.join(source_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)

        image_name, _ = image_file.split(".")
        magnitude_destination_path = os.path.join(magnitudes_path, f"{image_name}.npy")
        magnitude_spectrum = np.abs(dft_shift)

        np.save(magnitude_destination_path, magnitude_spectrum)
        phase_destination_path = os.path.join(phases_path, f"{image_name}.npy")
        phase_spectrum = np.angle(dft_shift)
        np.save(phase_destination_path, phase_spectrum)


def load_images(source_path, num_images):
    """
    Function to load the images from the given directory

    :param source_path: path to the directory with the images
    :param num_images: number of images to load
    :return: array of the loaded images
    """

    grayscale_images = []

    for i in tqdm(range(num_images)):
        pixels = load_img(os.path.join(source_path, f"image_{i}.jpg"), target_size=(128, 128))
        pixels = img_to_array(pixels)
        grayscale_images.append(pixels)

    return np.asarray(grayscale_images)


def load_phases_and_magnitudes(phases_path, magnitudes_path, num_images):
    """
    Function to load the phases and magnitudes from the given directories

    :param phases_path: path to the directory with the phases
    :param magnitudes_path: path to the directory with the magnitudes
    :param num_images: number of images to load
    :return: list of tuples of the loaded magnitudes and phases
    """

    magnitudes_and_phases = []

    for i in tqdm(range(num_images)):
        magnitude = np.load(os.path.join(magnitudes_path, f"image_{i}.npy"))
        phase = np.load(os.path.join(phases_path, f"image_{i}.npy"))
        magnitudes_and_phases.append((np.asarray(magnitude), np.asarray(phase)))

    return magnitudes_and_phases


def print_sample(grayscale_images, magnitudes_and_phases, num_images):
    """
    Function to print a sample of the grayscale images, magnitudes and phases

    :param grayscale_images: array of the grayscale images
    :param magnitudes_and_phases: list of tuples of the magnitudes and phases
    :param num_images: number of images to print
    """

    for i in range(num_images):
        pyplot.subplot(3, num_images, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(grayscale_images[i].astype("uint8"))

        pyplot.subplot(3, num_images, 1 + num_images + i)
        pyplot.axis("off")
        pyplot.imshow(np.log1p(magnitudes_and_phases[i][0]))

        pyplot.subplot(3, num_images, 1 + 2 * num_images + i)
        pyplot.axis("off")
        pyplot.imshow(magnitudes_and_phases[i][1])

    pyplot.show()
