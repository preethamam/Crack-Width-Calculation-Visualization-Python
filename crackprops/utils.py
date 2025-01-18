import numpy as np
from skimage import img_as_ubyte, img_as_float
from PIL import Image

class ImageManipulation:
    """
    A utility class for image manipulation.
    """

    @staticmethod
    def imoverlay(image, mask, color=[1, 1, 1]):
        """
        Create a mask-based image overlay.

        Args:
            image (np.ndarray): Input image, grayscale or RGB. Can be uint8, uint16, int16, float, or double.
            mask (np.ndarray): 2D binary mask (logical array).
            color (list): 1x3 list of RGB values in the range [0, 1].

        Returns:
            np.ndarray: Output RGB image with the mask locations colored.
        """
        mask = mask.astype(bool)

        if np.issubdtype(image.dtype, np.floating):
            image = img_as_ubyte(img_as_float(image))
        else:
            image = img_as_ubyte(image)

        color = np.array(color)
        if np.any((color < 0) | (color > 1)):
            raise ValueError("Color values must be in the range [0, 1].")

        color_uint8 = (color * 255).astype(np.uint8)

        if image.ndim == 2:
            out_red = image.copy()
            out_green = image.copy()
            out_blue = image.copy()
        elif image.ndim == 3 and image.shape[2] == 3:
            out_red = image[:, :, 0].copy()
            out_green = image[:, :, 1].copy()
            out_blue = image[:, :, 2].copy()
        else:
            raise ValueError("Input image must be either grayscale or RGB.")

        out_red[mask] = color_uint8[0]
        out_green[mask] = color_uint8[1]
        out_blue[mask] = color_uint8[2]

        return np.stack([out_red, out_green, out_blue], axis=-1)

class ImageUtils:
    """
    A utility class for saving images.
    """

    @staticmethod
    def save_image(image, filename):
        """
        Save an image to disk.

        Args:
            image (np.ndarray): Image to be saved.
            filename (str): Path to save the image.
        """
        im = Image.fromarray(image)
        im.save(filename)