import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def get_image_paths(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Error: {folder_path} is not a directory.")

    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith(".jpg")]
    return image_files


def rename_images(folder_path):
    image_files = get_image_paths(folder_path)

    for i, image_file in enumerate(tqdm(image_files, desc="Renaming images")):
        old_path = os.path.join(folder_path, image_file)
        new_name = f"image_{i}.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)


def convert_to_grayscale(source_folder, destination_folder):
    image_files = get_image_paths(source_folder)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for image_file in tqdm(image_files, desc="Converting to grayscale"):
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        image = Image.open(source_path)
        grayscale_image = image.convert("L")
        grayscale_image.save(destination_path)


def generate_phases(source_folder, destination_folder):
    image_files = get_image_paths(source_folder)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for image_file in tqdm(image_files, desc="Generating phase images"):
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)
        phase_spectrum = np.angle(dft_shift)
        phase_image = Image.fromarray(phase_spectrum, "L")
        phase_image.save(destination_path)


if __name__ == "__main__":
    color_path = "data/color"
    grayscale_path = "data/grayscale"
    phases_path = "data/phases"

    try:
        rename_images(color_path)
        convert_to_grayscale(color_path, grayscale_path)
        generate_phases(grayscale_path, phases_path)
    except ValueError as e:
        print(e)
