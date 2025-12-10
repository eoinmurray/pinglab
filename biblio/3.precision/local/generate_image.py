import numpy as np
from PIL import Image


def generate_image(size=16):
    # Create a 16x16 black image
    img_array = np.zeros((size, size), dtype=np.uint8)

    # Draw horizontal line
    img_array[size//2, :] = 255

    # Draw vertical line
    img_array[:, size//2] = 255

    return img_array