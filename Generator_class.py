import torch 

from torch import nn

class Generator(nn.Module):
  """
  The generator class.
  """

  def __init__(self, z_dim: int, img_channels: int, num_feature_maps: int, num_classes: int):

    """ Creates a Generator instance.

    Parameters
    ----------
    z_dim
        The dimension of the noise tensor (z).

    img_channels
        The number of channels in the generated image.

    num_feature_maps
        The number of feature maps carried through the generator. 

    num_classes
        The number of classes in the training data.

    """

    super(Generator, self).__init__()

    # Create an embedding of size z_dim for each class label
    self.embed = nn.Embedding(num_classes, z_dim)

    # The sequence of layers in the generator architecture:
    self.network = nn.Sequential(
      
        nn.ConvTranspose2d(z_dim + z_dim, num_feature_maps * 16, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_feature_maps * 16),
        nn.ReLU(),

        nn.ConvTranspose2d(num_feature_maps * 16, num_feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(num_feature_maps * 8),
        nn.ReLU(),

        nn.ConvTranspose2d(num_feature_maps * 8, num_feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(num_feature_maps * 4),
        nn.ReLU(),

        nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(num_feature_maps * 2),
        nn.ReLU(),

        nn.ConvTranspose2d(num_feature_maps * 2, img_channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
    )
      

  def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The forward propagation through the generator.

    Parameters
    ----------
    noise
        The latent vector of shape N x z_dim x 1 x 1.
        
    labels
        The label(s) of the generated image(s).


    Returns
    -------
    tensor
        The generated image(s) of shape N x 3 x 64 x 64.

    """

    # Create an embedding from the labels with the same size as z
    embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
    x = torch.cat([noise, embedding], dim=1)
 
    return self.network(x)