# Effect of Compression on Camera Identification via PRNU (Photo-Response Non-Uniformity)
This project implements a camera identification system using PRNU noise patterns on compressed and uncompressed images. It can evaluate the effect of compression from various schemes on an image image by comparing noise fingerprints extracted from images before and after compression.

## Features
PRNU noise extraction via wavelet denoising.

Fingerprint averaging from multiple images per camera.

Fast and robust camera matching using FFT-based normalized cross-correlation.

Supports JPEG, PNG, JPEG2000 (.jp2), and JPEG XL (.jxl) image formats.

Automatic validation and evaluation using a confusion matrix.

## Output
Prints per-image predictions with matching scores.

Displays a confusion matrix comparing predicted vs. actual camera sources.

## How It Works
Training Phase:

Extracts PRNU noise from each image.

Computes an average noise fingerprint for each camera.

Testing Phase:

Extracts noise from a test image.

Compares it against each camera fingerprint using normalized cross-correlation.

Selects the camera with the highest similarity score.

## Notes
Ensure images are grayscale or are converted to grayscale.

Noise extraction uses wavelet denoising (BayesShrink) from scikit-image.

JPEG XL and JP2 decoding handled via Pillow and pyjxl.

Let me know if you'd like a Markdown-formatted README.md file or additional CLI options like --train and --test paths.