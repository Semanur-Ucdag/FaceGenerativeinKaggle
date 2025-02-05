# FaceGenerativeinKaggle
Overview

This project implements an advanced Generative Adversarial Network (GAN) trained on the CelebA dataset. The model architecture consists of a Generator and a Critic (Discriminator), designed to generate high-quality synthetic images. The training process follows the Wasserstein GAN with Gradient Penalty (WGAN-GP) approach to improve stability and convergence.

Features

Custom Dataset Class: Loads and preprocesses images from the CelebA dataset.

Generator Model: Uses transposed convolutional layers with batch normalization and ReLU activations to generate realistic images.

Critic Model: Utilizes convolutional layers with instance normalization and LeakyReLU activations to distinguish real from fake images.

Gradient Penalty (WGAN-GP): Ensures training stability and prevents mode collapse.

Adaptive Training Strategy: Updates the critic multiple times per generator update to refine adversarial training.

Visualization Function: Displays generated images periodically during training.

Dependencies

This project requires the following libraries:torch, torchvision, numpy, matplotlib, PIL (Python Imaging Library), tqdm

Hyperparameters

Parameter/Value
Value =20
Batch Size = 32
Learning Rate = 1e-4
Latent Dimension (z_dim)= 175
Critic Cycles per Generator Update = 5

#Model Architecture

Generator

Uses transposed convolutions to upsample the latent vector.

Batch normalization ensures stable training.

Final layer applies Tanh activation to output RGB images.

Critic (Discriminator)

Uses convolutional layers with instance normalization.

LeakyReLU activation to allow gradient flow.

Predicts a real/fake score for each input image.

Dataset

The dataset used is CelebA, a large-scale dataset of celebrity faces. Images are resized to 128x128 and normalized to the range [0, 1].

Training Process

Critic Update:

Compute loss based on real and fake images.

Apply gradient penalty to enforce the Lipschitz constraint.

Update critic weights multiple times per generator update.

Generator Update:

Generate fake images from random noise.

Compute loss based on the critic’s prediction.

Optimize the generator to improve image realism.

Visualization & Model Saving:

Periodically visualize generated images.

Save trained model weights at the end of training.

Usage

To train the model, execute:

python train.py

To generate images using a trained model:

from generator import Generator
import torch

gen = Generator(z_dim=175)
gen.load_state_dict(torch.load('generator.pth'))
gen.eval()

z = torch.randn(8, 175)  # Generate 8 random noise samples
fake_images = gen(z)

Results

The model successfully generates high-quality human face images with realistic features. The WGAN-GP approach improves training stability and ensures better diversity in generated images.

License

This project is open-source and distributed under the MIT License.

