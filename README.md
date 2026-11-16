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