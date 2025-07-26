# Pix2Pix GAN - Satellite to Map Translation

This project implements the Pix2Pix Generative Adversarial Network (GAN) for image-to-image translation. Specifically, it learns to convert satellite imagery into corresponding map views using paired training data.

---

## Overview

Pix2Pix is a conditional GAN framework where the generator learns to map an input image to a target image, and the discriminator learns to distinguish between real and generated image pairs. In this case:

* **Input**: Satellite image (left half of the dataset image)
* **Target**: Map image (right half of the dataset image)

The model uses a U-Net-based generator and a PatchGAN discriminator as described in the original Pix2Pix paper.

---
## Example
<img width="927" height="653" alt="image" src="https://github.com/user-attachments/assets/82c294a5-0b51-4aef-839a-06f8fc5e2069" />

## Dataset

* Input images are expected to be of size `(256, 512, 3)`
* The dataset folder contains such paired images.
* Each image is split into:

  * `X`: Satellite image (`pixels[:, :256, :]`)
  * `Y`: Map image (`pixels[:, 256:, :]`)

---

## Preprocessing

* Images are loaded using Keras `load_img` and `img_to_array`
* All images are resized to `256x512`
* Normalized to range `[-1, 1]` to match the generator's `tanh` activation

---

## Model Architecture

### Generator

* A U-Net consisting of:

  * 8 encoder blocks (Conv2D + BatchNorm + LeakyReLU)
  * 7 decoder blocks (Conv2DTranspose + Dropout + Concatenate + ReLU)
  * Final layer: `tanh` activation to generate RGB image

### Discriminator

* PatchGAN architecture:

  * Takes a pair of images as input (source and target/generated)
  * Outputs a matrix of probabilities (real/fake per patch)
  * Uses sigmoid activation in the final layer

### GAN Model

* Combines the generator and discriminator
* Discriminator is frozen during GAN training
* Uses a weighted sum of adversarial loss (binary crossentropy) and L1 loss (mean absolute error)
* Loss function weights: `[1, 100]`

---

## Training

* Data is split into real and generated pairs for training
* For each batch:

  * Train discriminator on real and fake pairs
  * Train generator via the combined GAN model
* Logs batch-level training statistics: discriminator and generator losses
* Periodically saves:

  * Model checkpoints (`model_xxxxxx.h5`)
  * Visualizations of real vs generated outputs (`plot_xxxxxx.png`)

---

## Inference

* Load a trained generator model (e.g., `saved_model_10epochs.h5`)
* Feed a new satellite image and generate the corresponding map
* Plot and compare the source, generated, and target images

---

## Requirements

* Python 3.7+
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## Usage

```python
# Load and preprocess data
src_images, tar_images = load_images(DATASET_PATH)
data = preprocess_data([src_images, tar_images])

# Initialize models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

# Train
train(d_model, g_model, gan_model, data, n_epochs=100, n_batch=1)

# Test
model = load_model('saved_model_10epochs.h5')
plot_images(source_image, generated_image, target_image)
```

---

## Notes

* Training may take several hours depending on the dataset size and hardware
* Quality of output improves with more epochs and fine-tuned hyperparameters
* Generator outputs can be visually inspected to determine training progress

---


