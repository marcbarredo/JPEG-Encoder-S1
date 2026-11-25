# JPEG-Encoder-S1

## Description
This repository contains the implementation of Seminar 1, focused on image compression techniques, JPEG encoding, JPEG2000 and FFmpeg.

## Exercises
1. FFmpeg installation
2. RGB ↔ YUV conversion
3. Chroma subsampling
4. Image resizing with FFmpeg
5. Zigzag scanning
6. DCT and quantization
7. Run-length encoding
8. Mini-JPEG vs FFmpeg
9. JPEG vs JPEG2000
10. AI code review

## Technologies
- Python
- FFmpeg
- NumPy
- SciPy
- PyWavelets
- pytest

## Authors
Marc Barredo - U209162
Aran Bori - U232829


### Exercise 2 – RGB ↔ YUV Conversion

A `ColorTranslator` class was implemented to convert RGB colors to YUV and back using BT.601 coefficients.

The round-trip conversion was tested using RGB = (255, 100, 50). The recovered values were approximately (254.999, 99.999, 50.004).

Small numerical differences appeared due to floating-point precision and approximated transformation coefficients. However, after rounding to integer values, the original RGB color was successfully recovered.

**Conclusion:** The RGB ↔ YUV conversion works correctly for the tested color, with negligible reconstruction errors.

### Exercise 3 – Chroma Subsampling

The image was converted into four different chroma subsampling formats using FFmpeg: YUV 4:4:4, 4:2:2, 4:2:0 and 4:1:1.

| Format | Raw file size | Size reduction |
|---|---:|---:|
| 4:4:4 | 691,200 bytes | 0% |
| 4:2:2 | 460,800 bytes | 33.33% |
| 4:2:0 | 345,600 bytes | 50% |
| 4:1:1 | 345,600 bytes | 50% |

Despite the differences in file size, no significant visual differences were observed between the reconstructed images.

A second experiment was performed by applying Gaussian blur separately to the chrominance (U, V) and luma (Y) channels.

Blurring the chrominance channels produced only minor visual changes, while blurring the luma channel caused a noticeable loss of sharpness and texture.

**Conclusion:** JPEG and many video codecs use chroma subsampling because human vision is more sensitive to spatial detail in luminance than in chrominance. This allows the amount of color information to be reduced while preserving most of the perceived image quality.

### Exercise 4 – Image Resizing with FFmpeg

The original image (640 × 360 pixels) was resized using FFmpeg to three different resolutions: 320 × 180, 160 × 90 and 80 × 45.

A Python class called ImageResizer was implemented using the subprocess module to automate the resizing process.

Three scaling algorithms were compared: nearest neighbor, bilinear and bicubic.

Results:

Nearest neighbor: Produced more visible pixelation and irregular edges.

Bilinear: Generated smoother transitions, but some details appeared less defined.

Bicubic: Provided a good balance between smoothness and edge definition.

FFmpeg's libswscale normally uses bicubic interpolation by default when no alternative scaling algorithm is specified.

Conclusion: Reducing the image resolution decreases the amount of spatial information. The interpolation algorithm influences the visual appearance of the resized image, particularly edge sharpness, smoothness and aliasing artifacts.

### Exercise 5 – Zigzag Scanning

A `ZigzagEncoder` class was implemented to convert an 8×8 matrix into a sequence of 64 values following the JPEG zigzag scanning pattern.

The class includes two methods: `zigzag()` and `inverse_zigzag()`.

The implementation was tested using an 8×8 matrix containing values from 0 to 63. The inverse operation successfully reconstructed the original matrix without information loss.

**Conclusion:** Zigzag scanning orders DCT coefficients approximately from low to high spatial frequencies. After quantization, this arrangement tends to group zero-valued coefficients toward the end of the sequence, improving the efficiency of subsequent run-length encoding.

### Exercise 6 - DCT + quantization

The DCT transforms the 8x8 block of pixels into frequencies. The top-left corner represents average brightness and low frequencies, while the bottom-right corner represents fine details. 
We wrote the DCT by hand using matrix C and verified that it matches scipy.

Quantization is the only step that loses information: dividing by the table and rounding. The table has larger numbers at high frequencies, which the eye notices less.

The result table:
| Quality | Zeros (out of 64) | High-frequency zeros (out of 16) | Mean error |
|---|---:|---:|---:|
| 90 | 28 | 12 | 2.5 |
| 50 | 49 | 16 | 6.26 |
| 10 | 54 | 16 | 12.38 |

**Conclusion:** As the quality decreases, the high-frequency coefficients drop to zero (almost all of them at quality 50), which is why zigzag and RLE compress so well afterward. In return, the error increases. We can see it on the screenshot ex6_quantization.png