# Variational Autoencoder(VAE) for image reconstruction by the animation faces dataset
In this VAE example, I use five ConvNets for the encoder and decoder networks.

## Encoder
This defines the approximate posterior distribution ![](http://latex.codecogs.com/svg.latex?q(z|x)), which takes as input an observation and outputs a set of parameters for specifying the conditional distribution of the latent representation ![](http://latex.codecogs.com/svg.latex?z). In this example, simply model the distribution as a diagonal Gaussian, and the network outputs the mean and log-variance parameters of a factorized Gaussian. Output log-variance instead of the variance directly for numerical stability.
## Decoder
This defines the conditional distribution of the observation ![](http://latex.codecogs.com/svg.latex?q(x|z)), which takes a latent sample  as input and outputs the parameters for a conditional distribution of the observation. Model the latent distribution prior ![](http://latex.codecogs.com/svg.latex?p(z)) as a unit Gaussian.
