
import cv2
import re
import os
import pathlib

import pandas as pd
import numpy as np


IMAGE_DIR_PATH = pathlib.Path(os.path.join(os.path.dirname(__file__), '../data/images'))
PHASE_DIR_PATH = pathlib.Path(os.path.join(os.path.dirname(__file__), '../data/phases'))
METATADA_PATH = pathlib.Path(os.path.join(os.path.dirname(__file__), '../data/metadata.csv'))


class DataGen:
    def __init__(self):
        self.metadata = self.get_metadata()

    def list_images_in_path(self, path):
        image_types = ['.jpg', '.jpeg', '.png']
        image_list = []
        for _, _, files in os.walk(path):
            for file in files:
                if file.endswith(tuple(image_types)):
                    image_list.append(file)
        return image_list

    def image_to_phase(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(dft)
        phase_spectrum = np.angle(dft_shift)
        return phase_spectrum

    def get_metadata(self):
        type_pattern = re.compile(r'\.jpg|\.jpeg|\.png')
        if not pathlib.Path(METATADA_PATH).exists():
            image_list = self.list_images_in_path(IMAGE_DIR_PATH)
            metadata = pd.DataFrame(columns=['image_paths', 'phase_paths'])
            metadata['image_paths'] = image_list
            metadata['phase_paths'] = metadata['image_paths'].apply(
                lambda x: type_pattern.sub('.npy', x)
            )
            metadata.to_csv(pathlib.Path(METATADA_PATH))
            return metadata
        else:
            # search for images that are not in metadata and add them
            image_list = self.list_images_in_path(IMAGE_DIR_PATH)
            metadata = pd.read_csv(pathlib.Path(METATADA_PATH))
            metadata_image_list = metadata['image_paths'].tolist()
            paths = []

            for image in image_list:
                if image not in metadata_image_list:
                    paths.append({'image_paths': image, 'phase_paths': type_pattern.sub('.npy', image)})
            paths = pd.DataFrame(paths)
            metadata = pd.concat([metadata, paths])
            return metadata

    def gen_phases(self):
        # Generate phase images for all images if not already generated
        for image_path, phase_path in zip(self.metadata['image_paths'], self.metadata['phase_paths']):
            if not pathlib.Path(os.path.join(PHASE_DIR_PATH, phase_path)).exists():
                phase = self.image_to_phase(os.path.join(IMAGE_DIR_PATH, image_path))
                np.save(os.path.join(PHASE_DIR_PATH, phase_path), phase)


if __name__ == '__main__':
    dg = DataGen()
    dg.gen_phases()
