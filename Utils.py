"""
    This module contains utility functions that are used to train the conditional DCGAN.
"""


import torch
from torch import nn
import math
import matplotlib.pyplot as plt 
import numpy as np
from Discriminator_class import Discriminator
from Generator_class import Generator
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage import io, color, util, measure
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
 



def plot_images(images: torch.Tensor, labels: list):
    """A function that displays images and their class labels on a grid using matplotlib.

    Parameters
    ----------
    images
        A list of image tensors.
        
    labels
        A list of string class labels associated with the images.
    """

    # Set the number of rows and columns in the grid
    n_cols = 5 if len(images) > 5 else len(images)
    n_rows = math.ceil(len(images) / 5)

    plt.figure(figsize=(n_cols, n_rows))

    for index, image in enumerate(images, 1):
        image = image / 2 + 0.5     # denormalize the image
        plt.subplot(n_rows, n_cols, index)
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.title(labels[index - 1])
        plt.axis("off")

    plt.tight_layout()
    plt.show()




def initialise_weights(model):
    """A function that initialises the weights.
    It applies the weight initialisation used in the DCGAN paper - https://arxiv.org/pdf/1511.06434.pdf
    All weights are initialized from a zero-centered Normal distribution with standard deviation 0.02.

    Parameters
    ----------
    model
        The model instance whose weights will be initialised.
    """

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)




def test_implementation(num_samples:int, img_channels:int, img_size:int, num_classes:int, noise_dim:int, labels:int, disc_num_feature_maps:int, gen_num_feature_maps:int) -> str:
  """Tests the generator and discriminator classes and whether they output the correct sizes.
  
  Parameters
    ----------
    num_samples
        Size of the batch of images. 
    
    img_channels
        Number of channels in the training and the generated images.
        
    img_size
        The width and height of the training and generated images.
        
    num_classes
        The number of classes in the training data.
        
    noise_dim
        The dimension of the noise vector.
        
    labels
        The labels of the batch images.
        
    disc_num_feature_maps
        The number of feature maps propagated through the discriminator.
        
    gen_num_feature_maps
        The number of feature maps carried through the generator.
        

  Returns
  -------
  str
      A string saying if the implementation of the generator and discriminator are correct or not.
  """

  # Create an image tensor with random pixel values
  x = torch.randn(num_samples, img_channels, img_size, img_size)
  discriminator = Discriminator(img_channels, disc_num_feature_maps, num_classes, img_size)
  initialise_weights(discriminator)
  
  # Test if the discriminator returns the correct output
  assert discriminator(x, labels).shape == (num_samples, 1, 1, 1), "Discriminator outputs wrong size"

  
  # Create a random noise vector
  z = torch.randn((num_samples, noise_dim, 1, 1))
  generator = Generator(noise_dim, img_channels, gen_num_feature_maps, num_classes)
  initialise_weights(generator)

  # Test if the generator returns the correct output
  assert generator(z, labels).shape == (num_samples, img_channels, img_size, img_size), "Generator outputs wrong size"

  # If none of the above asserts fail, the generator and discriminator architectures were implemented successfully.
  return "The generator and discriminator architectures were implemented successfully"




def calculate_CID_indices(z_dim:int, generator:Generator, training_images:dict, num_fake_imgs:int) -> dict:
  """Calculates the creativity, inheritance and diversity indices.
  
  Parameters
    ----------
    z_dim
        The dimension of the noise from which the generator generates images. 

    generator
        The generator model. 
    
    training_images
        A dictionary containing all the training images grouped by their labels, {label_index: list_of_images}

    num_fake_imgs
        The number of generated images to evaluate
   
  Returns
  -------
  dict
      A dictionary containg creativity, inheritance and diversity indices for each label.
  """

  CID_indices = {}
  device = next(generator.parameters()).device

  # *********** Calculate the CID index for each class ***********
  for label in training_images:
    real_imgs = training_images[label]

    # Generate images of this class
    noise = torch.randn(num_fake_imgs, z_dim, 1, 1).to(device)
    labels = torch.LongTensor(np.repeat(label, num_fake_imgs)).to(device)
    generated_imgs = generator(noise, labels)
    # Denormalise and convert the generated tensors to numpy arrays
    generated_imgs = [(np.transpose(np.array(x.data.cpu()), (1, 2, 0)) / 2) + 0.5  for x in generated_imgs]
    
    
    # *************** Creativity Index ***************
    #   1) Remove all the generated images that are similar to the training images (i.e., ssim >= 0.8)
    Grem = list(filter(lambda fake: all(ssim(real, fake, channel_axis=2) < 0.8 for real in real_imgs), generated_imgs))

    #   2) Creativity index is the percentage of the remaining images
    creativity = len(Grem) / len(generated_imgs)


    # *************** Inheritance Index *************** 
    #   1) Convert the images to grayscale and uint8 format
    gray_real_imgs = list(map(lambda img: util.img_as_ubyte(rgb2gray(img)), real_imgs))  
    gray_fake_imgs = list(map(lambda img: util.img_as_ubyte(rgb2gray(img)), Grem)) 
    
    #   2) Calculate the Gray Level Co-occurrence of each image
    #   3) Take the average of GLCM-contrast values of all images
    gcr = np.mean([graycoprops(graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True), 'contrast') for img in gray_real_imgs])
    gcf = np.mean([graycoprops(graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True), 'contrast') for img in gray_fake_imgs])

    #   4) Inheritance = 1 − (|gcr−gcf| / max{gcr,gcf})  
    inheritance = 1 - (abs(gcr - gcf) / max(gcr, gcf))


    # *************** Diversity Index ***************
    #   1) Group to the same cluster all generated images that have an SSIM >= 0.8
    clusters = []
    fake_imgs = generated_imgs
    while len(fake_imgs) > 0:
      img1 = fake_imgs.pop(0)
      cluster = [img1]
      for i, img2 in enumerate(fake_imgs):
        if ssim(img1, img2, channel_axis=2) >= 0.8:
          cluster.append(fake_imgs.pop(i))
      clusters.append(cluster)

    #   2) Diversity = −∑ (pi log(pi)) ; pi = |Ci|/|Grem|  
    diversity = -1 * sum(list(map(lambda x: x * math.log(x), [len(c) / len(Grem) for c in clusters])))

    
    # Save the creativity, inheritance and diversity indices of this label.
    CID_indices[label] = [creativity, inheritance, diversity]

  return CID_indices
 