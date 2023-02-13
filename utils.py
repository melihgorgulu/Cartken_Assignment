import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List


def get_binary_map(img_path: str) -> np.array:
    pil_image = Image.open(img_path).convert('1')
    binary_img = np.asarray(pil_image, dtype=np.uint8)
    binary_img = np.logical_not(binary_img).astype(int)
    return binary_img


def vis_binary_map(b_map: np.array):
    plt.imshow(b_map, cmap='Greys', interpolation='nearest')
    plt.title("Boundry Map")
    plt.show()


def vis_segments(segments: List, start_idx, end_index):
    assert end_index >= start_idx, "start index should be equal or lower from end_index"
    img_to_vis = np.zeros(shape=segments[0].shape, dtype=np.uint8)
    for idx in range(start_idx, end_index + 1):
        img_to_vis = np.logical_or(img_to_vis, segments[idx])
    vis_binary_map(img_to_vis)


def vis_important_segments(segments: List):
    # Important segments 11, 33 ,46
    segm_11, segm_33, segm_46 = segments[11], segments[33], segments[46]
    fig, ax = plt.subplots(1)
    ax.imshow(segm_11, cmap='Reds', interpolation='none')
    ax.imshow(segm_33, cmap='Blues', interpolation='none', alpha=0.5)
    ax.imshow(segm_46, cmap='Greens', interpolation='none', alpha=0.5)
    ax.set_title('Different Boundry Segments')
    ax.text(10, 25, 'Seg32:Red, Seg11:Blue, Seg46:Green', bbox={'facecolor': 'white', 'pad': 5})
    plt.show()

