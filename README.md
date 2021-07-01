# Variational-Autoencoder-VAE-for-image-reconstruction-by-the-animation-faces-dataset

In this VAE example, I use five ConvNets for the encoder and decoder networks. 
## Encoder
This defines the approximate posterior distribution $q(z|x)$, which takes as input an observation and outputs a set of parameters for specifying the conditional distribution of the latent representation $z$. In this example, simply model the distribution as a diagonal Gaussian, and the network outputs the mean and log-variance parameters of a factorized Gaussian. Output log-variance instead of the variance directly for numerical stability.
## Decoder
This defines the conditional distribution of the observation $q(x|z)$, which takes a latent sample  as input and outputs the parameters for a conditional distribution of the observation. Model the latent distribution prior $p(z)$ as a unit Gaussian.
## Reparameterization trick
To generate a sample $z$ for the decoder during training, you can sample from the latent distribution defined by the parameters outputted by the encoder, given an input observation $x$. However, this sampling operation creates a bottleneck because backpropagation cannot flow through a random node.

To address this, use a reparameterization trick. In our example, you approximate $z$ using the decoder parameters and another parameter $\epsilon$ as follows:

$$z = \mu + \sigma \times \epsilon$$\
where $\mu$ and $\sigma$ represent the mean and standard deviation of a Gaussian distribution respectively. They can be derived from the decoder output. The $\epsilon$ can be thought of as a random noise used to maintain stochasticity of $z$. Generate $\epsilon$ from a standard normal distribution.

The latent variable $z$ is now generated by a function of $\mu$, $\sigma$ and $\epsilon$, which would enable the model to backpropagate gradients in the encoder through $\mu$ and $\sigma$ respectively, while maintaining stochasticity through $\epsilon$.
## Loss function
Original distribution $P(x)$:
$$P(x) = \int_{z} P(z)P(x|z)$$
And we hope $P(x)$ the bigger the better, then
$$\text{Maximum} \ L = \sum_{x} \text{log}P(x)$$
where
$$ \text{log}P(x) =  \int_{z} q(z|x)\text{log}P(x)$$
$$=\int_{z} q(z|x)\text{log}\left \(\frac{P(z,x)}{P(z|x)}\right )$$
$$=\int_{z} q(z|x)\text{log}\left \(\frac{P(z,x)}{q(z|x)}\frac{q(z|x)}{P(z|x)}\right )$$
$$=\int_{z} q(z|x)\text{log}\left \(\frac{P(z,x)}{q(z|x)}\right ) + \int_{z} q(z|x)\text{log}\left \(\frac{q(z|x)}{P(z|x)}\right )$$
$$=\int_{z} q(z|x)\text{log}\left \(\frac{P(z,x)}{q(z|x)}\right ) + KL \left \(q(z|x)||P(z|x)\right )$$
The second term of the above formula is a value greater than or equal to 0, so we found a lower bound of $\text{log}P(x)$
$$\text{log}P(x) \geq \int_{z} q(z|x)\text{log}\left \(\frac{P(z,x)}{q(z|x)}\right )$$

VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood:

$$\text{log} p(x) \geq \text{ELBO} = \text{E}_{q(z|x)}\left \[ \text{log}\frac{p(x,z)}{q(z|x)} \right ]$$
