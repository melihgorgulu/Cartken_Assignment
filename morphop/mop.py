import numpy as np
from scipy.ndimage import label
from typing import List


class MorphOps:
    def __init__(self, kernel_size: int = 3, pad_image: bool = True):
        """

        :param kernel_size: structure kernel size
        :param pad_image: pad image or not
        """
        self.pad_image = pad_image
        self.kernel_size = kernel_size
        self.structuring_kernel = np.full(shape=(kernel_size, kernel_size), fill_value=1, dtype=np.uint8)
        self.flat_sub_matrices = None
        self.cached = False
        self.original_shape = None

    def calculate_flat_sub_matrices(self, img):
        self.original_shape = img.shape

        if self.pad_image:
            pad_width = self.kernel_size - 2
            img = np.pad(array=img, pad_width=pad_width, mode='constant')
            padded_shape = img.shape

        else:
            padded_shape = self.original_shape

        row_reduce = padded_shape[0] - self.original_shape[0]
        col_reduce = padded_shape[1] - self.original_shape[1]
        self.flat_sub_matrices = np.asarray([img[i:(i + self.kernel_size), j:(j + self.kernel_size)]
                                             for i in range(padded_shape[0] - row_reduce)
                                             for j in range(padded_shape[1] - col_reduce)], dtype=np.uint8)
        self.cached = True

    def erode_image(self, img) -> np.array:
        """
        :param img: input binary map
        :return: eroded image
        """
        if self.cached:
            img_eroded = np.array([1 if (i == self.structuring_kernel).all() else 0 for i in self.flat_sub_matrices],
                                  dtype=np.uint8)
            img_eroded = np.reshape(img_eroded, newshape=self.original_shape)
        else:
            self.calculate_flat_sub_matrices(img)
            img_eroded = np.array([1 if (i == self.structuring_kernel).all() else 0 for i in self.flat_sub_matrices],
                                  dtype=np.uint8)
            img_eroded = np.reshape(img_eroded, newshape=self.original_shape)
        return img_eroded

    def dilate_image(self, img) -> np.array:
        """
        :param img: input binary map
        :return: dilated image
        """
        if self.cached:
            img_dilated = np.array([1 if (i == self.structuring_kernel).any() else 0 for i in self.flat_sub_matrices],
                                   dtype=np.uint8)
            img_dilated = np.reshape(img_dilated, newshape=self.original_shape)
        else:
            self.calculate_flat_sub_matrices(img)
            img_dilated = np.array([1 if (i == self.structuring_kernel).any() else 0 for i in self.flat_sub_matrices],
                                   dtype=np.uint8)
            img_dilated = np.reshape(img_dilated, newshape=self.original_shape)

        return img_dilated

    def find_boundries(self, img) -> np.array:
        """
        :param img: binary map
        :return: numpy array that contains boundries
        """
        eroded_image = self.erode_image(img)
        dilated_image = self.dilate_image(img)
        inner_boundry = img - eroded_image
        outer_boundry = dilated_image - img
        # fuse boundries
        boundry_image = np.bitwise_xor(inner_boundry, outer_boundry)
        return boundry_image

    @staticmethod
    def extract_boundry_segments(boundry_map: np.array) -> List:
        """

        :param boundry_map: boundries binary map
        :return: List of segments, each item of list is numpy array
        """
        segmented_boundries, n_features = label(boundry_map, structure=np.ones(shape=(3, 3)))

        all_labels = np.unique(segmented_boundries)[1:]
        segments = []
        for lbl in all_labels:
            # take current segment
            current_segment = np.where(segmented_boundries != lbl, 0, segmented_boundries)
            # set 1 for labeled places
            current_segment[current_segment == lbl] = 1
            segments.append(current_segment)
        return segments
