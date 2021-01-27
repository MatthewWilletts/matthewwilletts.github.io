---
layout: post
title: Improving VAEs' Robustness to Adversarial Attacks -- Part 1
---


This is part one of a two part series where I explain in some detail the ideas in my recent ICLR paper `Improving VAEs' Robustness to Adversarial Attacks', work done with my friends and colleagues at the University of Oxford and The Alan Turing Institute.

If you are not familiar with Variational Autoencoders, I recommend you start with either of the review articles [Doesch, (2016)](https://arxiv.org/abs/1606.05908) or [Kingma & Welling, (2019)](https://arxiv.org/abs/1906.02691).

## Adversarial Attacks

First, some background on adversarial attacks.
What are they are why do they matter?
Machine Learning algorithms are being rolled out to perform ever more automated decision-making at scale.
To those with ulterior motives, manipulating the decisions made by these algorithms might be very appealing -- by hoodwinking these systems you could trick them into giving you the prediction or decision that you desire, not that which the model would normally give.
Adversarial attacks are exactly these attempts, to make a model do what an adversarial agent wants, _not_ what the model would do if unmolested.
$$\renewcommand{\vector}[1]{\boldsymbol{\mathbf{#1}}}$$
$$\renewcommand{\v}{\vector}$$
$$\newcommand\vv[1]{\vec{\v{#1}}}$$
$$\newcommand\KL{\mathrm{KL}}$$
$$\newcommand\ELBO{\mathcal{L}}$$
$$\newcommand\expect{\mathbb{E}}$$
$$\newcommand\argmin{\mathrm{arg \,min}}$$

It turns out you can fool neural network classifiers quite easily, with very small distortions that are almost imperceptible to the human eye.
Szegedy et al., (2013) showed that adding a well-chosen, small-magnitude, distortion to an input image can make a deep learning classifier give the wrong predicted label for the image.
More than that, the adversary can fool the classifier into labelling the image to their _chosen_ class with very high confidence.
This research direction continues at pace, studying how best to attack neural networks, why they are vulnerable, and how they can be defended.

|  Neural Network Classifiers under attack |
|:-:|
|![Classifer Attack](/images/seatbelt/attack_plots/classifier_attacks/negative1.png#center){:class="img-responsive"}
[Left] Original inputs [Middle] Distortion [Right] Distorted Inputs|
|The images on the left and right recieve completely different classifications, the rightmost column are all confidently classified as ‘Ostrich’.|
|Reproduced from _Szegedy et al., (2013)_|

Recall that deep learning classifiers generally output predictions via a _logit_ representation -- the unnormalised log probabilties of the predicted class label.
The networks actual prediction is thus the class with maximum probability, so the prediction is index of the entry in the vector of logits that has the largest value.
Its confidence will be greater the larger the differece is between that largest value and the rest.
So, to fool a classifier into outputing the adversary's chosen prediction, the name of the game is diminishing the logit value of the true class and increasing the logit value of the target class.

The most standard attacks on deep learning models assume _white box_ access -- the adversary has access to the models architecture and parameters.
This means that the distortion can be found by the adversary by simple optimisation -- find the distortion that maximally boosts the chosen logit output, while also trying to keep the distortion's magnitude relatively small (either by penalising it to a chosen degree or constraining it to fall within some range).

## Adversarial Attacks on Deep Generative Models

Now, neural networks are not just used to make classifiers.
There are other kinds of deep learning model.
As reviewed in [Doesch, (2016)](https://arxiv.org/abs/1606.05908) and [Kingma & Welling, (2019)](https://arxiv.org/abs/1906.02691), we can use neural networks, along with some ideas from Bayesian statistics, to build models _for_ the data we have.
So instead of learning conditional distributions -- such as a classifier where we train the model to give us $$p_\theta(y\mid \v x)$$, the probability over classes $$y$$ _given_ an input data $$\v x$$ for model parameters $$\theta$$ -- instead we want a model that assigns a probability to the input itself -- so we want to obtain $$p_\theta(\v x)$$ for our data, or something approximate to it.
This is an example of _unsupervised learning_.

Models of this type are called _generative models_, and those that use neural networks as components are called Deep Generative Models (DGMs).
Variational Autoencoders (VAEs) (Kingma & Welling, 2014; Rezende et al., 2014) are a particular, very popular, class of these.
VAEs are most commonly applied to image data.
Other than a few toy experiments we will be concerned with images too.

As dicussed in Doesch, (2016) and Kingma & Welling, (2019), VAEs are _latent variable models_: data is modelled as having been generated from a (small) number of latent (i.e., unobserved) highly explanatory quantities that between them neatly encapsulate the important facts about a given datapoint.
These models are useful as they enable us to explain, to compress, our datapoints using their inferred latent representatations, and also as they enable us to generate realistic `synthetic data'.

VAEs are made of two halfs: first an _encoder_ that maps input data to a distribution over latent variables and second a _decoder_ that maps settings of the latent variables to a conditional distribution over data-space.
As the number of latent variables is commonly much smaller than the dimensionality of the data, taking a datapoint through the model -- encoding and decoding it -- means we are `bottlenecking' our data through the latent space.
Roughly speaking, this bottlenecking means that we are asking our model to learn a parsimonious set of representations that preserves the information in the original input data as best it can.

The encoder's output is often used for downstream tasks, for example text analysis (Xu et al., 2017), for modelling molecules in drug discovery (Kusner et al., 2017), for image compression (Theis et al., 2017; Townsend et al., 2019), and as the perceptual part of a reinforcement learning system (Ha & Schmidhuber, 2018; Higgins et al., 2017b).
VAEs are even themselves used to protect classifiers from adversarial attack (Schott et al., 2019; Ghosh et al., 2019)

All this leads to a natural question: can we fool a VAE?
This will not be quite the same as attacking a classifier -- after all, we are not considering a class label, as VAEs are an example of unsupervised learning.
And if so, how can we try to ensure that VAEs are robust to adversarial attacks against them?
As the fundamental mode of operation of a VAE is that of encoding an input datapoint to a low-dimensional latent space then reconstructing it, fooling a VAE means manipulating the input so that it reconstructs a distorted input to a chosen _target image_.
Attacks of this nature have already been proposed.
But before we describe them, and then our way to help defend against them, we need just a little bit more on the set-up of VAEs.


#### Quick Recap of VAEs

To ground the rest of this post, let's choose some notation and go into a little bit of technical background.

We have a dataset $$\mathcal{D}=\{\v{x}_i\}_{i=1,...,N}$$, each datapoint $$\v x \in \mathcal{X}=\mathbb{R}^{d_x}$$.
We model each datapoint as having been generated from a latent variable $$\v z \in \mathcal{Z}=\mathbb{R}^{d_z}$$, with $$d_z \ll d_x$$.
A VAE's encoder provides us with each datapoint's approximate posterior $$q_\phi(\v z \mid \v x)$$ -- we can apply our neural network encoder to a given datapoint $$\v x$$ and get the statistical parameters for its posterior.
The standard choice in VAEs is $$q_\phi(\v z \mid \v x)=\mathcal{N}(\v z \mid \v{\mu}=\v{\mu}_\phi(\v x),\v{\sigma}=\v{\sigma}_\phi(\v x))$$, where $$\v{\mu}_\phi(\cdot)$$ and $$\v{\sigma}_\phi(\cdot)$$ are neural networks that return the mean and standard deviation respectively (and we are imposing diagonal covariance for the per-datapoint approximate posterior).
The likelihood is $$p_\theta(\v x \mid \v z)$$.
Again, for standard VAEs with continuous data, a common choice is $$p_\theta(\v x \mid \v z)=\mathcal{N}(\v x \mid \v{\mu}=\v{\mu}_\theta(\v z),\v{\sigma}=\v{\sigma}_\mathcal{X})$$ where $$\v{\mu}_\theta(\cdot)$$ is the neural network that returns the mean, i.e. the decoder, and $$\v{\sigma}_\mathcal{X}$$ is a chosen, fixed, hyperparameter.

Standard VAEs' have $$p(\v z)=\mathcal{N}(\v z \mid \v{\mu}=\v 0,\v{\sigma}=\v 1)$$ as the prior over their latents, and are trained to maximise the Evidence Lower Bound over our dataset $$\mathcal{D}=\{\v{x}_i\}_{i=1,...,N}$$,

$$\log p_\theta(\mathcal{D}) \geq \ELBO(\mathcal{D}; \theta,\phi) = \frac{1}{N}\sum_{i=1}^N
\left[\expect_{\v{z}\sim q_\phi(\v z \mid \v{x}_i)}\left[\log p_\theta(\v{x}_i \mid \v z)\right] - \KL \left( q_\phi(\v z \mid \v{x}_i) \mid\mid p(\v z)\right)\right]$$

where $$\KL(\cdot\mid\mid\cdot)$$ is the [Kullbach-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition).
This optimisation is commonly done using minibatches of data, which provides an unbiased sampler of the objective.

#### Attacking VAEs via the Latent Space

Now, manipulating an image to fool a classifier means tricking the model into reducing one output (the true label's logit value) and increasing another one (the target label's logit value).
For a VAE, we are trying to change ALL the pixels (for colour images, RGB sub-pixels) of the image to take the particular values they take in the target image.
MNIST has 784 greyscale pixels.
32 by 32 pixel colour images have 3072 subpixels.
So this is a lot of values to manipulate -- as one might expect the most effective attack methods developed so far for VAEs attempt to sidestep this problem.

Remember that the latent space is of much lower dimensionality that the data itself, which makes the latent space sound like a potentially promising avenue of attack.
Instead of directly trying to tune the output of the model, the reconstruction, to be close to the chosen target, the adversary can instead try to match in the latent space.

Reconstructions are done taking samples from a datapoint's latent representation.
If the attacked (i.e. distorted) datapoint gives rise to the same latent representation as the target image, then the two reconstructions will be the same (up to the noise of the sampling procedure) and the attack has been successful.
So there has a been a small shift in the nature of the attack -- in truth we are aiming for the attacked point to reconstruct to the _reconstuction_ of the target image.

So, how to go about it?
The adversary wants to find some (ideally small magnitude) distortion $$\v d$$ to add to the initial input $$\v x$$ such that the distorted input $$\v{x}^d = \v x + \v d$$ will reconstruct to be close to the chosen target $$\v{x}^t$$, up to the fidelity of the model, **but we want to find the attack using the low dimensional latent representations**.
In Gondim-Ribeiro et al., (2018) and Tabacof et al., (2016), an effective way of attacking VAEs is to match the distorted inputs posterior representations to the target's, using the $$\KL$$ as an objective,

$$\v{d} = \argmin_{\v{d}^{\prime}} \left[\KL\left( q_\phi(\v z \mid \v x + \v{d}^{\prime} ) \mid\mid q_\phi(\v z \mid \v{x}^t) \right) + \lambda \mid\mid \v{d}^{\prime} \mid\mid_2 \right]$$

where we have added an $$L_2$$ penalty on the distortion, weighted by $$\lambda$$, to encourage lower-magnitude attacks.
After all, the trivial attack $$\v{x}^d=\v{x} + \v{d}=\v{x}^t$$ would exactly match latent representations, but doesn't correspond to _fooling_ the model -- so we want $$\v d $$ to be small.

This method works -- VAEs can be fooled by attacks constructed this way.

|| VAES under latent-space attack ||
|:-:|:-:|:-:|
|![VAE Latent Attack](/images/seatbelt/attack_plots/celeba_original.png#center){:height='300px'}|![VAE Latent Attack](/images/seatbelt/attack_plots/celeba_vae.png#center){:class="img-responsive" height='300px'}|![VAE Latent Attack](/images/seatbelt/attack_plots/celeba_target.png#center){:class="img-responsive" height='300px'}|
|Original inputs |Original Recon. - Adv. Input - Adv. Recon. - Target Recon.|Target Images|
||Here the adversary is trying to get find a distortion to the left column images so they reconstruct via the latent representations of the target, so their recontructions are similar.||
||Reproduced from _Gondim-Ribeiro et al., (2018)_||

#### Why can this latent attack be done?

Why is it that attacks of this type work?
Well, they work because different input points can have very similar posteriors.
Ok, so why does that happen?

Let's start by thinking about what happens when we attack a standard deep auto-encoder, rather that a VAE.
That means reconstructions are made entirely deterministically: we map a data point to its hidden, bottlenecked, representation from just a forward pass through a neural network, and then we pass that determinisitic value through the decoder to get our reconstruction.
In training such a model we are not explicitly demanding that nearby points in the hidden space decode to similar points in data space -- it can do perfectly well learning a lookup-table.

For sufficiently powerful and flexible neural networks there is no particular reason why nearby embedded points should decode to look much like each other.
For inputs not in the training set, and for points in the latent space that are not embedding points, the properties of the model are pretty free-form -- there hasn't been a direct constraint on what the model should do when faced with them.
The mappings can be intrinsically non-smooth: small changes in input can lead to large changes in latent representation, and small changes in latent representation can lead to large changes in reconstruction.
Any structure that does come about will be from the inductive biases of the neural networks used.

This limiting case of a lookup table is also exactly what we get from a VAE with vanishing posterior noise.
Variational inference has a regularising effect as the per-datapoint posteriors are each driven to be close to the prior by the $$\KL$$ term in the ELBO.
Deviations from that have to be paid for by increases in reconstruction fidelity -- increases in the log likelihood term in the ELBO.
However, that VAEs are attackable tells us that in VAE training as usually specified the model is similar enough to a lookup-table to enable attacks.

## Defending VAEs

A plausible solution that presented itself to us would be to control the noise of the posteriors in the VAE's latent space, the degree of noise in the learnt posteriors.
Let's think about why this will help.
Latents in a VAE are sampled from posteriors with non-zero variance.
It is the noisiness of latent representations that rewards similar data points being mapped to similar latent representations.
The noise in the latent representations means that there is intrinsic uncertainty in what value the sampled latent representation produced by a given datapoint will take.
The encoder thus is rewarded for putting similar inputs to nearby areas of the latent space, hedging against that uncertainty.

A regular VAE is attackable, yes, but we can still expect it to be more robust than a deterministic autoencoder due to the learnt noise in their posterior representations.
What we found, and describe in more detail below, is that _by increasing the noise in our data points' posteriors we can make VAEs more robust to attack_.

In considering the noise of a VAE's latent space we need to consider the _aggregate posterior_.
It is simply the mixture we obtain from embedding all our data in the latent space:

$$q_\phi(\v z; \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N}q_\phi(\v z \mid \v{x}_i)$$

How much the different components in this mixture _overlap_ is the key geometric consideration we latch onto to give VAEs increased robustness.

But how best to do this?
One could do this by controlling and constraining the neural implementation of the underlying networks in the model.
Instead, here we are going to do this by upweighting various penalisation terms in the training objective, that have the effect of rewarding models that have increased overlap.
After all, if we think insufficient `overlap' is the relevant concept here, then it makes sense to develop methods of defense that aim to increase it.
Even within this general approach, there are different ways one could do this.


#### Tuning the noise in VAEs

All methods to obtain increased overlap that add extra regularisation to the training objective can be expected to reduce model performance, as measured by the value reached by the un-regularised ELBO after training.
But different methods will have different effects on the model, so have different tradeoffs associated with them.

The methods we discuss here were outlined first in the literature as ways of obtaining `disentangled' representations -- representations that are in some sense one-for-one with the important aspects of variation that exist in the training data.
For photos of simple objects, cubes and spheres of various sizes and colours against a white background, say, a disentangled model would hopefully learn latent representations that had one axis in the latent space corresponding to shape, one to size, a few for position in space, some for rotations in space, and perhaps a small number for colour, saturation and lighting.

It turns out that disentangling requires more than just applying the methods of regularisation we will discuss.
As found by Locatello et al., (2019) and Rolinek et al., (2019), precise structuring of models, careful hyperparameter tuning and a fair degree of luck is needed to obtain disentangled representations with these methods.
For our purposes, however, this problem does not matter:
increasing the various regularisation terms we will discuss does reliably lead to increased overlap.
Interestingly, the adversarial robustness we obtain does not seem to be correlated with the degree of disentangling we observe in the trained models.


$$\mathbf{\beta}\textbf{-VAEs}$$:

The simplest way to increase the noise in the latent representations is to increase the strength of the $$\KL$$ penalisation already present in the ELBO.
This approach gives us a $$\beta$$-VAE (Higgins et al., 2017a); now the training objective is

$$\ELBO_{\beta}(\mathcal{D}, \beta; \theta,\phi) = \frac{1}{N}\sum_{i=1}^N
\left[\expect_{\v{z}\sim q_\phi(\v z \mid \v{x}_i)}\left[\log p_\theta(\v{x}_i \mid \v z)\right] - \beta \KL \left( q_\phi(\v z \mid \v{x}_i) \mid\mid p(\v z)\right)\right]$$

where $$\beta>1$$.
This method does increase overlap: all per-datapoint posteriors are being driven to be equal to the prior, which means that information is being quite forcefully lost as $$\beta$$ increases.
As a result, for the increases in overlap it gives you it has quite a negative impact on model performance.
In part this comes naturally from the fact that this penalisation is per-datapoint, so much like in a vanilla VAE the coordination that takes place in determining how the latent space fits together is an emergent phenomena.

$$\mathbf{\beta}\textbf{-TCVAEs}$$:

Instead we can try to improve the overlap in the latent space in a way that itself depends on the aggegrate posterior -- after all the property we are trying to obtain is a global property of the model, so it makes sense to give the model access to its own global properties during training.
That means that the per-datapoint posteriors would be able to interact with each other through the training objective directly, improving the ability of the model to coordinate how the aggregate posterior is laid out.

We are trying to increase the symmetry of the latent space -- a smooth, gap-less aggregate posterior is highly symmetric.
One form of symmetry we can have in multivariate probability distributions is statistical independence -- that a distribution is equal to the product of its marginals.
The discepancy between a distribution and the product of its marginals can be captured by the _total correlation_ (Watanabe, 1960), the $$\KL$$ between these two entities:
$$ \mathrm{TC}(p(\v{a})) = \KL\left(p(\v a) \mid\mid \prod_{j=1}^{d} p(a_j) \right)$$
where $$d$$ is the dimensionality of the variable at hand.
So, we can upweight the total correlation of the aggregate posterior, then, as a way to induce model robustness.

Unlike upweighting $$\KL \left( q_\phi(\v z \mid \v{x}) \mid\mid p(\v z)\right)$$ as in a $$\beta$$-VAE, upweighting this constraint applied to the aggregate posterior does not in itself impose a particular length scale, nor any particular shape.
At the risk of stating the obvious, example of distributions with zero total correlation include: an aggregate posterior that is a product of uniform distributions, one that is a product of standard Laplace distributions, one that is a product of standard Gaussian distributions, one that is a product of Gaussian distributions with scalar standard deviation $$\sigma$$, and one that is a product of some mix of these options.

(Perhaps this makes it obvious why this divergance is one of the fundamental training objectives in Independent Components Analysis (Everson & Roberts, 2001) where we want to learn statistically-independent latent representations, and in the context of VAEs this idea was first put to use with the aim of finding clean latent representations where each axis describes one factor of variation of the data.)

By asking for a mutually independent aggregate posterior we are asking for a folding symmetry.
That if we ‘fold over’ the aggregate posterior along any of the axes in the latent space, the posterior matches itself when laid back down.
Any (off-axis) holes in the aggegate posterior violate this, so the model is rewarded for ‘filling in’ the aggegate posterior at those places.
(The mirror of this is true for any (off-axis) peaks in the aggregate posterior -- the model is rewarded for smoothing them out.)

If we penalise the total correlation of the aggregate posterior in a VAE we get a $$\beta$$-TCVAE (Chen et al., 2018).
Interestingly this total correlation term can be revealed to be already present in the standard VAE ELBO's $$\KL$$ term.
So rather than adding a totally new mathematical construct into the mix, in a $$\beta$$-TCVAE we are in fact strengthening an already present, but weak, tendency.
The $$\beta$$-TCVAE training objective is

$$\begin{align}\ELBO_{\beta\mathrm{-TC}}(\mathcal{D}, \beta ; \theta,\phi) = &\frac{1}{N}\sum_{i=1}^N
\left[\expect_{\v{z}\sim q_\phi(\v z \mid \v{x}_i)}\left[\log p_\theta(\v{x}_i \mid \v z)\right] - \KL \left( q_\phi(\v z \mid \v{x}_i) \mid\mid p(\v z)\right)\right] \nonumber \\
&-(\beta-1)\KL\left(q_\phi(\v z;\mathcal{D}) \mid\mid \prod_{j=1}^{d_z} q_\phi(z_j;\mathcal{D})\right)
\end{align}$$

where the factor is $$(\beta-1)$$, not $$\beta$$, because as said above there is already one times the aggregate posterior's total correlation ‘hidden’ in $$\KL \left( q_\phi(\v z \mid \v{x}) \mid\mid p(\v z)\right)$$.

#### What does this do to the latent space?

Just to see how things work in low dimensions, we train some models, a vanilla VAE, a $$\beta$$-VAE and a $$\beta$$-TCVAE, with 2D latent spaces, on the toy ‘swiss-roll’ dataset with varying values of $$\beta$$.


||VAE $$q(\v z)$$| VAE recons|
|:-:|:-:|:-:|
||![VAE q](/images/seatbelt/swiss_plots/vae_1.png#center){:height='100px'}|![VAE q](/images/seatbelt/swiss_plots/vae_1_roll.png#center){:height='100px'}|

||$$\beta$$-VAE $$q(\v z)$$| $$\beta$$-VAE recons|$$\beta$$-TCVAE $$q(\v z)$$|$$\beta$$-TCVAE recons|
|:-:|:-:|:-:|:-:|:-:|
|$$\mathbf{\beta=8}$$|![B-VAE q](/images/seatbelt/swiss_plots/bvae_8.png#center){:height='100px'}|![B-VAE recons](/images/seatbelt/swiss_plots/bvae_8_roll.png#center){:height='100px'}|![B-TCVAE q](/images/seatbelt/swiss_plots/tcvae_8.png#center){:height='100px'}|![B-TCVAE recons](/images/seatbelt/swiss_plots/tcvae_8_roll.png#center){:height='100px'}|
|$$\mathbf{\beta=32}$$|![B-VAE q](/images/seatbelt/swiss_plots/bvae_32.png#center){:height='100px'}|![B-VAE recons](/images/seatbelt/swiss_plots/bvae_32_roll.png#center){:height='100px'}|![B-TCVAE q](/images/seatbelt/swiss_plots/tcvae_32.png#center){:height='100px'}|![B-TCVAE recons](/images/seatbelt/swiss_plots/tcvae_32_roll.png#center){:height='100px'}|
|$$\mathbf{\beta=128}$$|![B-VAE q](/images/seatbelt/swiss_plots/bvae_128.png#center){:height='100px'}|![B-VAE recons](/images/seatbelt/swiss_plots/bvae_128_roll.png#center){:height='100px'}|![B-TCVAE q](/images/seatbelt/swiss_plots/tcvae_128.png#center){:height='100px'}|![B-TCVAE recons](/images/seatbelt/swiss_plots/tcvae_128_roll.png#center){:height='100px'}|

These plots show the aggregate posterior and the reconstructions (the modes of the likelihood conditioned on a sample of each per-datapoint posterior).
Clearly the amount of overlap increases with $$\beta$$ for both kinds of model, but the $$\beta$$-TCVAEs appear to do this in a more structured way (as we would hope) and, unlike the $$\beta$$-VAE, they not suffer from such catastrophic degradation in model quality for large $$\beta$$ -- for large $$\beta$$ the $$\beta$$-VAE's reconstructions have collpased to a line.

Going from toy to real data, we measure this effect quantitiatively for models trained on CelebA.

||Effect of regularisation on overlap and model quality on $$\beta$$-VAEs and $$\beta$$-TCVAEs||
|:-:|:-:|:-:|
|![Overlap](/images/seatbelt/main_plots/bvae_btc_overlap_sigma.png#center){:class="img" width="auto" height="200" style="display:block; max-height:200px; max-width:400px; height:100%;}|![Likelihood](/images/seatbelt/main_plots/bvae_btc_likelihood_deg.png#center){:class="img" width="auto" height="200" style="display:block; max-height:200px; max-width:400px; height:100%;}|![ELBO](/images/seatbelt/main_plots/bvae_btc_elbo_deg.png#center){:class="img" width="auto" height="200" style="display:block; max-height:200px; max-width:400px; height:100%;}|
|Density plot of $$\mid\mid\v{\sigma}_\phi(\v{x})\mid\mid_2$$|$$\log p_{\theta}(\v{x}\mid\v{z})$$  as a fn. of $$\beta$$ | ELBO as a fn. of $$\beta$$.|

The left plot shows a KDE of the norm of the encoder standard deviation for a VAE, a $$\beta$$-VAE and a $$\beta$$-TCVAE each trained on CelebA, $$\beta=10$$.
The $$\beta$$-VAE's posterior variance saturates, while the $$\beta$$-TCVAE's does not, and as such is able to induce more overlap.
In the centre and right plots we show the log likelihood and ELBO for both model types as a function of $$\beta$$.
Clearly the model quality degrades to a lesser degree for the TC-penalised models under increasing $$\beta$$. 

That TC penalisation can be applied without causing as strong a degradation in model quality compared to that found in $$\beta$$-VAEs (as measured by the fidelity of reconstructions and the value of the $$\beta=1$$ ELBO), while still increasing overlap, is the key reason we think people should use it to `robustify' VAEs when needed.
There is not much point having a robust model if the model is now strongly degraded.


## Adversarial Attacks on $$\beta$$-TCVAEs

Well, as you might expect, the adversary has a harder time finding useful pathologies in the latent structures of $$\beta$$-TCVAE than in vanilla VAEs.
We can get a first handle on this by measuring the value reached of the adversaries attack objective, as a function of $$\beta$$.

||Adversarial loss for $$\beta$$-TCVAEs for various datasets||
|:-:|:-:|:-:|
|![dSprites](/images/seatbelt/main_plots/dsprites_shallow_adv_loss_beta_cropped.png#center){:class="img" width="auto" height="200" style="display:block; max-height:200px; max-width:400px; height:100%;}|![Chairs](/images/seatbelt/main_plots/chairs_shallow_adv_loss_beta.png#center){:class="img" width="auto" height="200" style="display:block; max-height:200px; max-width:400px; height:100%;}|![Faces](/images/seatbelt/main_plots/faces_shallow_adv_loss_beta.png#center){:class="img" width="auto" height="200" style="display:block; max-height:200px; max-width:400px; height:100%;}|
|dSprites|Chairs|Faces|

Higher loss indicates more robustness.
Note that the loss axis is logarithmic. 
Shading corresponds to the 95% CI produced by attacking 20 images for each combination of $$d_{\v{z}}=\{4,8, 16,32,64,128\}$$.
Following the attack methods in Gondim-Ribeiro et al., (2018) and Tabacof et al., (2016), we have taken 50 geometrically distributed values of $$\lambda$$ between $$2^{-20}$$ and $$2^{20}$$ (giving $$1000$$ total trials).
$$\beta > 1$$ clearly induces a much larger loss for the adversary relative to $$\beta = 1$$ for all the datasets we studied.

This increased robustness also holds for other attacks: both for attacking the VAE via its output (trying directly to manipulate the reconstruction to be like the target) and also for a novel attack we proposed based on the 2-Wasserstein distance.

This is all very promising.
It's neat that we can re-purpose methods first proposed for a totally differnet purpose to reliably enhance the robustness of these models.
But, we can do better than this.
If we take these ideas and apply them to hierarchical VAEs -- VAEs with layers of latent variables -- we can obtain VAEs that are more robust still to adversarial attack, and also have an even better robustness vs. model quality tradeoff than $$\beta$$-TCVAEs.
But for that, we need to look to Part 2...


#### References

**Ricky T Q Chen, Xuechen Li, Roger Grosse, and David Duvenaud**. _Isolating Sources of Disentanglement in Variational Autoencoders_. (arXiv:1802.04942) In NeurIPS, 2018.

**Carl Doersch**. _Tutorial on Variational Autoencoders_. arXiv:1606.05908, 2016

**Richard Everson & Stephen J Roberts**. _Independent Component Analysis_. Cambridge University Press., 2001

**Partha Ghosh, Arpan Losalka, and Michael J Black**. _Resisting Adversarial Attacks Using Gaussian Mixture Variational Autoencoders__. (arXiv:1806.00081) In AAAI, 2019

**George Gondim-Ribeiro, Pedro Tabacof, and Eduardo Valle**. _Adversarial Attacks on Variational Autoencoders_. arXiv:1806.04646, 2018

**David Ha and Jürgen Schmidhuber**. _World Models_. arXiv:1803.10122, 2018

**Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed and Alexander Lerchner**. _$$\beta$$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework_. In ICLR, 2017a

**Irina Higgins, Arka Pal, Andrei A. Rusu, Loic Matthey, Christopher P Burgess, Alexander Pritzel, Matthew Botvinick, Charles Blundell, and Alexander Lerchner**. _DARLA: Improving Zero-Shot Transfer in Reinforcement Learning_. (arXiv:1707.08475) In ICML, 2017b

**Diederik P Kingma and Max Welling**. _Auto-encoding Variational Bayes_. (arXiv:1312.6114) In ICLR, 2014

**Diederik P Kingma and Max Welling**. _An Introduction to Variational Autoencoders_. Foundations and Trends in Machine Learning 12-4:307-392 (arXiv:1906.02691), 2019

**Matt J. Kusner, Brooks Paige, José Miguel Hernández-Lobato**. _Grammar Variational Autoencoder_. (arXiv:1703.01925) in ICML, 2017

**Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf and Olivier Bachem**. _Challenging common assumptions in the unsupervised learning of disentangled representations_. (arXiv:1811.12359) in ICML, 2019

**Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra**. _Stochastic Backpropagation and Approximate Inference in Deep Generative Models_. (arXiv:1401.4082) In ICML, 2014

**Michal Rolinek, Dominik Zietlow,and Georg Martius**. _Variational Autoencoders Pursue PCA Directions (by Accident)_. (arXiv:1812.06775) In CVPR, 2019

**Lukas Schott, Jonas Rauber, Matthias Bethge, and Wieland Brendel**. _Towards the first adversarially robust neural network model on MNIST_. (arXiv:1805.09190) In ICLR, 2019

**Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus**. _Intriguing properties of neural networks_. (arXiv:1312.6199) In ICLR, 2014.

**Pedro Tabacof, Julia Tavares, and Eduardo Valle**. _Adversarial Images for Variational Autoencoders_. (arXiv:1612.00155), In NeurIPS Workshop on Adversarial Training, 2016

**Lucas Theis, Wenzhe Shi, Andrew Cunningham and Ferenc Huszár**. _Lossy Image Compression with Compressive Autoencoders_. (arXiv:1703.00395) In ICLR, 2017

**James Townsend, Thomas Bird, and David Barber**, _Practical Lossless Compression with Latent Variables using Bits Back Coding_. (arXiv:1901.04866) In ICLR, 2019

**Satosi Watanabe**. _Information Theoretical Analysis of Multivariate Correlation_. IBM Journal of Research and Development, 4(1):66–82, 1960

**Weidi Xu, Haoze Sun, Chao Deng, and Ying Tan**. _Variational Autoencoder for Semi-Supervised Text Classification_. (arXiv:1603.02514) In AAAI, 2017