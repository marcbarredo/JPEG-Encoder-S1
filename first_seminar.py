
import cv2
import numpy as np
import matplotlib.pyplot as plt

##EXERCISE 1##
class ColorTranslator:
    """
    A class to convert colors between RGB and YUV color spaces
    using the BT.601 standard.
    """

    def rgb_to_yuv(self, r, g, b):
        """
        Convert an RGB color to YUV.

        Args:
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).

        Returns:
            A tuple containing the Y, U, and V components.
        """

        # Calculate the luma (brightness) component
        y = 0.299 * r + 0.587 * g + 0.114 * b

        # Calculate the chrominance (color difference) components
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b

        return y, u, v

    def yuv_to_rgb(self, y, u, v):
        """
        Convert a YUV color back to RGB.

        Args:
            y: Luma component.
            u: Blue chrominance component.
            v: Red chrominance component.

        Returns:
            A tuple containing the R, G, and B components.
        """

        # Apply the inverse YUV to RGB transformation
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u

        return r, g, b




def compare_luma_chroma_blur(image_path):
    """
    Compare the visual effects of blurring the luma channel
    versus blurring the chroma channels.
    """

    # Load the original image and convert it from BGR to YUV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Create two independent copies of the YUV image
    chroma_blurred = yuv.copy()
    luma_blurred = yuv.copy()

    # Apply Gaussian blur only to the chroma channels (U and V)
    chroma_blurred[:, :, 1] = cv2.GaussianBlur(
        yuv[:, :, 1], (31, 31), 0
    )
    chroma_blurred[:, :, 2] = cv2.GaussianBlur(
        yuv[:, :, 2], (31, 31), 0
    )

    # Apply Gaussian blur only to the luma channel (Y)
    luma_blurred[:, :, 0] = cv2.GaussianBlur(
        yuv[:, :, 0], (31, 31), 0
    )

    # Convert the processed images back to RGB for visualization
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chroma_rgb = cv2.cvtColor(chroma_blurred, cv2.COLOR_YUV2RGB)
    luma_rgb = cv2.cvtColor(luma_blurred, cv2.COLOR_YUV2RGB)

    # Display the three images side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [original_rgb, chroma_rgb, luma_rgb]
    titles = ["Original", "Chroma blur (U, V)", "Luma blur (Y)"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Save the results for the report
    cv2.imwrite(
        "images/chroma_blurred.png",
        cv2.cvtColor(chroma_rgb, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        "images/luma_blurred.png",
        cv2.cvtColor(luma_rgb, cv2.COLOR_RGB2BGR)
    )


# Run the experiment
compare_luma_chroma_blur("images/original.png")


if __name__ == "__main__":

    # Create an instance of the ColorTranslator class
    translator = ColorTranslator()

    # Define an example RGB color
    rgb_original = (255, 100, 50)

    # Convert the original RGB color to YUV
    yuv = translator.rgb_to_yuv(*rgb_original)

    # Convert the YUV color back to RGB
    rgb_recovered = translator.yuv_to_rgb(*yuv)

    # Display the original and converted color values
    print("Original RGB:", rgb_original)
    print("Converted YUV:", yuv)
    print("Recovered RGB:", rgb_recovered)

    # Round the recovered RGB values to the nearest integer
    rgb_rounded = tuple(round(value) for value in rgb_recovered)

    # Display the rounded result
    print("Rounded RGB:", rgb_rounded)

    # Check whether the original RGB values were recovered
    print("Exact reconstruction:", rgb_original == rgb_rounded)