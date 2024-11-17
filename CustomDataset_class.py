from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Union


class CustomDataset(Dataset):
    """
       The CustomDataset class inherited from the torch Dataset class
    """
    def __init__(self, img_paths: list, label_index: int, label_name: str, transform=None):
      """Creates a CustomDataset instance.

      Parameters
      ----------
      img_paths
          A list of the full paths to images.

      label_index
          The numeric label of the images.

      label_name
          The name of the class label of the images.

      transform (optional)
          The transformation(s) to be appied to the images.

      """

      self.img_paths = img_paths
      self.label_index = label_index
      self.label_name = label_name
      self.transform = transform


    def __len__(self) -> int:
      """Counts the number of images in a batch.

      Returns
      -------
      int
          The number of images in the batch.

      """
      return len(self.img_paths)


    def __getitem__(self, index: int) -> Union[torch.Tensor, int, str]:
      """Retreives an image with its label at a specific index.

      Parameters
      ----------
      index
          The index of the image to be returned.

      Returns
      -------
      Union[torch.Tensor, int, str]
          The image, label, and class name.

      """
      # Load the image and apply the transformation(s) to it.
      img_path = self.img_paths[index]
      image = Image.open(img_path) #.convert('RGB')

      if self.transform is not None:
          image = self.transform(image)

      # Return the image tensor and the label
      return image, self.label_index, self.label_name