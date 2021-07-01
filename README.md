# Variational-Autoencoder-VAE-for-image-reconstruction-by-the-animation-faces-dataset

In this VAE example, I use five ConvNets for the encoder and decoder networks. 
## Encoder
This defines the approximate posterior distribution $q(z|x)$, which takes as input an observation and outputs a set of parameters for specifying the conditional distribution of the latent representation $z$. In this example, simply model the distribution as a diagonal Gaussian, and the network outputs the mean and log-variance parameters of a factorized Gaussian. Output log-variance instead of the variance directly for numerical stability.
## Decoder
This defines the conditional distribution of the observation $q(x|z)$, which takes a latent sample  as input and outputs the parameters for a conditional distribution of the observation. Model the latent distribution prior $p(z)$ as a unit Gaussian.
## Reparameterization trick
To generate a sample $z$ for the decoder during training, you can sample from the latent distribution defined by the parameters outputted by the encoder, given an input observation $x$. However, this sampling operation creates a bottleneck because backpropagation cannot flow through a random node.

To address this, use a reparameterization trick. In our example, you approximate $z$ using the decoder parameters and another parameter $\epsilon$ as follows:
$z = \mu + \sigma \times \epsilon$
