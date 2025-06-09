# 25Spring_Machine_Learning
Here's my code for the final competition of CSCI-SHU 360 Machine Learning in NYUSH 25Spring.

Description: Representation Learning
Train your autoencoder to learn representations of the fusion Pokemon images. Every image is of shape (3, 128, 128). Every channel (r, g, or b ) pixel has a value between 0 and 1 (normalized from 0 to 255). The reconstructed image will also be compared against the same 0 to 1 pixels to compute the MSE. Constraints: the max bottleneck representation generated using the encoder is of 8192 dimensions (flattened). Your model needs to finish inference on 3.3k images within 50 secs. The score is reconstruction_error / probing_acc (lower the better).
