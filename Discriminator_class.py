import torch
from torch import nn

class Discriminator(nn.Module):
  """
  The discriminator class.
  """
  def __init__(self, in_channels: int, num_feature_maps: int, num_classes: int, img_size: int):
    """ Creates a Discriminator instance.

    Parameters
    ----------
    in_channels
        The number of channels in the training and generated images.

    num_feature_maps
        The number of feature maps propagated through the discriminator.

    num_classes
        The number of classes in the training data.

    img_size
        The width and height of the training and the generated images

    """

    super(Discriminator, self).__init__()

    self.img_size = img_size

    # Create an embedding with the same size as the image for each class label
    self.embed = nn.Embedding(num_classes, img_size * img_size)

    # The sequence of layers in the discriminator architecture:
    self.network = nn.Sequential(
        nn.Conv2d(in_channels + 1, num_feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2),

        nn.Conv2d(num_feature_maps, num_feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(num_feature_maps * 2),
        nn.LeakyReLU(0.2),

        nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(num_feature_maps * 4),
        nn.LeakyReLU(0.2),

        nn.Conv2d(num_feature_maps * 4, num_feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(num_feature_maps * 8),
        nn.LeakyReLU(0.2),

        nn.Conv2d(num_feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),
        nn.Sigmoid()
    )


  def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The forward propagation through the discriminator.

    Parameters
    ----------
    x
        The image(s) to be classified as real or fake.
        
    labels
        The label(s) associated with the image(s).
        

    Returns
    -------
    tensor
        A probability that the image is real or fake.

    """
    
    embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
    x = torch.cat([x, embedding], dim=1)

    return self.network(x)