from typing import Tuple, Union
import os
import numpy as np
import cv2
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100

from nn_lib.data import Dataset


class Cifar100Dataset(Dataset):
    def __init__(self, train=True, **kwargs):
        """
        Create a dataset
        :param train: train or test images would be loaded
        """
        path = os.path.dirname(os.path.abspath(__file__)) + '/data'
        self.data_set = CIFAR100(root=path, train=train, download=True)

    def _stretch(self, image):
        return np.asarray(image).flatten() / 255.

    def _onehot_encode(self, label):
        result = np.zeros(100)
        result[label] = 1
        return result

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        result = self.data_set[index]
        result = self._stretch(result[0]), self._onehot_encode(result[1])
        return result

    def __len__(self) -> int:
        return len(self.data_set)

    def _resize_image(self, image, width, height):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def visualize(self, predictions: Union[np.ndarray, None] = None) -> None:
        width = 480
        height = 480
        index = 0
        dataset_size = len(predictions)

        def on_click(event):
            nonlocal index
            if event.button is MouseButton.LEFT:
                plt.cla()
                index = index + 1

                if index == dataset_size:
                    index = 0

                tensor, id = self.data_set[index]
                image = np.asarray(tensor)
                plt.title(f"id: {id} predicted {predictions[index]}")
                plt.imshow(image)

        plt.connect('button_press_event', on_click)
        plt.imshow(np.asarray(self.data_set[index][0]))
        plt.show()


if __name__ == '__main__':
    print(Cifar100Dataset()[0])
