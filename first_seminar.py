
import cv2
import numpy as np
import matplotlib.pyplot as plt

import subprocess
from pathlib import Path

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

class ImageResizer:
    """
    A class to resize images using FFmpeg through Python subprocess.
    """

    def __init__(self, input_path):
        # Store the path of the original image
        self.input_path = str(input_path)

    def resize(self, width, height, output_path, algorithm=None):
        """
        Resize an image using FFmpeg.

        Args:
            width: Target image width in pixels.
            height: Target image height in pixels.
            output_path: Path where the resized image will be saved.
            algorithm: Optional scaling algorithm (e.g. bilinear, bicubic).

        Returns:
            None.
        """

        # Build the FFmpeg command
        command = [
            "ffmpeg",
            "-y",
            "-i", self.input_path,
        ]

        # Select a scaling algorithm if specified
        if algorithm is not None:
            command.extend(["-sws_flags", algorithm])

        # Set the target resolution
        command.extend([
            "-vf", f"scale={width}:{height}",
            output_path
        ])

        # Execute FFmpeg and raise an error if it fails
        subprocess.run(command, check=True)

        # Confirm that the output was generated
        print(f"Image saved: {output_path}")


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

class ZigzagEncoder:
    """
    A class to convert an 8x8 matrix into a zigzag sequence
    and reconstruct the original matrix.
    """

    def zigzag(self, matrix):
        """
        Convert an 8x8 matrix into a list of 64 values
        following the JPEG zigzag scanning order.
        """

        # Check that the input matrix has the correct dimensions
        if len(matrix) != 8 or any(len(row) != 8 for row in matrix):
            raise ValueError("Input matrix must be 8x8")

        result = []

        # There are 15 diagonals in an 8x8 matrix
        for diagonal in range(15):

            # Find the coordinates belonging to each diagonal
            positions = []

            for row in range(8):
                col = diagonal - row

                if 0 <= col < 8:
                    positions.append((row, col))

            # Reverse every even diagonal to create the zigzag pattern
            if diagonal % 2 == 0:
                positions.reverse()

            # Extract the matrix values in zigzag order
            for row, col in positions:
                result.append(matrix[row][col])

        return result

    def inverse_zigzag(self, sequence):
        """
        Reconstruct an 8x8 matrix from a zigzag sequence
        containing 64 values.
        """

        # Validate the sequence length
        if len(sequence) != 64:
            raise ValueError("Input sequence must contain 64 values")

        # Create an empty 8x8 matrix
        matrix = [[0 for _ in range(8)] for _ in range(8)]

        index = 0

        # Visit the diagonals in the same order as the encoder
        for diagonal in range(15):

            positions = []

            for row in range(8):
                col = diagonal - row

                if 0 <= col < 8:
                    positions.append((row, col))

            if diagonal % 2 == 0:
                positions.reverse()

            # Place each value back into its original position
            for row, col in positions:
                matrix[row][col] = sequence[index]
                index += 1

        return matrix


if __name__ == "__main__":

    # EXERCISE 3: Luma vs chroma blur
    compare_luma_chroma_blur("images/original.png")

    # EXERCISE 2: RGB to YUV conversion

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

    # EXERCISE 4: Resize images with FFmpeg

    # Create an instance of the ImageResizer class
    resizer = ImageResizer("images/original.png")

    # Define the target resolutions
    resolutions = [
        (320, 180, "half"),
        (160, 90, "quarter"),
        (80, 45, "eighth")
    ]

    # Generate resized images automatically
    for width, height, name in resolutions:
        resizer.resize(
            width,
            height,
            f"images/python_resized_{name}.png"
        )

    # Compare different scaling algorithms
    algorithms = ["neighbor", "bilinear", "bicubic"]

    # Resize the image using each scaling algorithm
    for algorithm in algorithms:
        resizer.resize(
            80,
            45,
            f"images/resize_{algorithm}.png",
            algorithm=algorithm
        )

    # EXERCISE 5: Zigzag scanning

    # Create an instance of the ZigzagEncoder class
    zigzag_encoder = ZigzagEncoder()

    # Create an example 8x8 matrix
    original_matrix = [
        [row * 8 + col for col in range(8)]
        for row in range(8)
    ]

    # Convert the matrix into a zigzag sequence
    zigzag_sequence = zigzag_encoder.zigzag(original_matrix)

    # Reconstruct the original matrix
    recovered_matrix = zigzag_encoder.inverse_zigzag(zigzag_sequence)

    # Display the results
    print("\nOriginal matrix:")
    for row in original_matrix:
        print(row)

    print("\nZigzag sequence:")
    print(zigzag_sequence)

    print("\nRecovered matrix:")
    for row in recovered_matrix:
        print(row)

    # Check whether the original matrix was recovered
    print("\nExact reconstruction:",
          original_matrix == recovered_matrix)