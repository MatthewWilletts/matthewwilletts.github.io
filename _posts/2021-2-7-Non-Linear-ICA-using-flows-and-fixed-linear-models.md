---
layout: post
title: Learning Independent Representations using Flows & <i>Fixed</i> Linear Models
---


This post gives an overview of the ideas in my upcoming [AISTATS 2021 paper](https://arxiv.org/abs/2002.07766) `Learning Bijective Feature Maps for Linear ICA', work done with my friends and colleagues at the University of Oxford, UCL, and The Alan Turing Institute.


One branch of ML research aims to build models that learn descriptions of data that are _'disentangled'_.
We want the model to pull-apart input data, commonly images, into the separate and clearly meaningful factors that explain it, ie that cogently and cleanly summarise it.
We want to do this just from looking at the input data, the images themselves -- so with no external additional source of information.

Commonly, to obtain disentangled representations we look to find a set of descriptive factors that are in a technical sense _independent_ of each other (though there are other approaches out there too).
An example that helps us get a handle on this idea would be learning 'disentangled' representations for a stack of photos of peoples' faces.
We can describe a face via a set of 'atomic' facts.
Perhaps some given photo is of a: _'smiling'_ _'young'_ _'man'_ _'with red hair'_ _'wearing sunglasses'_, _'against a green background'_.
It is reasonable to expect that _'wearing sunglasses'_ is pretty independent from _'having red hair'_, as (probably) is being _'young'_ and so on down the list[^1].

By specifiying a model that tries to find these independent, simple, compressed, descriptions of data, the hope is that the individual learnt factors each correspond to separate and interpretable aspects of the data.

An approach that was popular for this problem, $$\beta$$--TCVAEs (Chen et al., 2018) and related models, where we train generative model for our data with an extra independence-between-learnt-representations term penalisation term, has turned out to be highly dependent on careful tuning of the process -- in other words you have to be pretty lucky for it to work (Locatello et al., 2019; Rolinek et al., 2019).
See [my post on robustness in VAEs](https://matthewwilletts.github.io/Defending-VAEs-from-Adversarial-Attack/) for (much) more discussion of these methods and what they can _actually_ be useful for. 

Our paper gives a new way to obtain _statistically independent_ latent representations of high-dimensional data like images, by bringing together some new ideas in ML with both ideas from random projections and the classical method of _indendependent components analysis_.

## Background: Independent Components Analysis

<!-- Perhaps you've heard of Principle Components Analysis (PCA), where we use some gorgeous linear algebra to find a simple low-dimensional representation of our data.
 -->

_Skip this section if you already know about linear Independent Components Analysis._

Before we get to all that, a brief skim over how we can attack this problem, of trying to learn statistically independent representations, when we only have to work with low-dimensional data.
It turns out that often linear methods are enough, and being linear they have all sorts of pleasant properties.
This linear approach to the problem will be a key sub-module in our full approach.

While Principle Components Analsis (PCA) is the blockbuster algorithm for discovering latent structure in data using linear algebra, we are interested in its sister method, the often-overlooked but arguably much cooler method _Independent Components Analysis_ (ICA).
$$\renewcommand{\vector}[1]{\boldsymbol{\mathbf{#1}}}$$
$$\renewcommand{\v}{\vector}$$
$$\newcommand\vv[1]{\vec{\v{#1}}}$$
$$\newcommand\KL{\mathrm{KL}}$$
$$\newcommand\ELBO{\mathcal{L}}$$
$$\newcommand\expect{\mathbb{E}}$$
$$\newcommand\diff{\mathrm{d}}$$
$$\newcommand\argmin{\mathrm{arg \,min}}$$

In ICA we want to find latent factors that explain the data, but unlike in PCA we are going to require that our discovered latent variables are statistically independent, instead of merely just explanatory of the variance in the data.

The classic applied problem that ICA solves is _blind source separation_: there are some _sources_ of audio, say, that are all making sounds on top of each other, a group of musicians playing intruments in an band, for example, and we have laid out a bunch of microphones randomly all around the room so we end up with a set of simultaneously-made audio recordings.
We want to _reverse_ the mixing process, and obtain separate recordings of each pure source, i.e. one for each intrument.
We want to _unmixing_ our data.

Say we have $$d_x$$ microphones, so we have a $$d_x$$ recordings of the room.
Each of those microphones will have picked up a slightly different local audio signal -- one happened to be really near the guy playing a tuba, one was off away in a corner, and so on.
<!-- So, we put down $$d_x$$ microphones.
 --><!-- Now we will make an assumption that makes things a fair bit simpler mathematically: that we picked $$d_$$ microphones because we knew that there were $$d_x$$ different instruments being played.
 -->

Now we can start to describe ICA.
Our multi-track audio data is a time series of $$N$$ temporal segments.
Let's denote each momentary microphone recording data as $$\v{x}_{ji}$$, $$j\in\{1,...,d_x\}$$, $$i\in\{1,...,N\}$$, which all together form the data matrix $$\v X$$.
So the first index $$i$$ is over time, and $$j$$ indexes over the different microphones we have.
The simple version of ICA, and the one we will build off, really doesn't care about the temporal order of the data -- we can randomly shuffle the order that our $$N$$ datapoints appear and nothing would change.
In order words, our approach works for independent and identically distributed (iid) data -- and it turns out we can view these momentary snippets of audio recordings as a set of iid observations for the purpose of this model -- so $$i$$ can be viewed as just an index over datapoints.

We then make a (hopefully reasonable) modelling choice, the claim, that, taken overall, there is no dependency between different intruments being player together or apart within any particular time-slice across our recordings.
To continue with the musical setting of this problem, we are saying we are model our data as being a recording of the freest-poosible jazz improv -- whether or not any instrument is being played at any particular moment is independent of what other intruments are being played.
We will denote the pure slices of waveform for the instruments as $$s_{ki}$$, $$k\in\{1,...,d_s\}$$, $$i\in\{1,...,N\}$$.
Again $$i$$ indexes over time (or a matched shuffling of it), and $$k$$ indexes over the $$d_s$$ different intruments ($$d_s\leq d_x$$).
It is $$\v{S}$$, the whole matrix of the latent (ie, unobserved) intrument waveforms that we want to infer.

Then we are going to make another modelling assumption, that the process by which the audio sources were _mixed_ by the natural environment and then recorded at each microphone was _instantaneous, linear, and constant over time_.
Instantaneous means we won't worry about different instruments have different lags for different microphones -- these would be present if the room was really big so the effect of some instruments being really far from some microphones but very near to others was large enough that we have to take it into account.
Linear means that each recording is the result of a simple weighted sum of the sources, and being constant over time means that the mixing does not depend on the time index $$i$$.
Thus the recorded signal can be written:

$$x_{j,i} = a_{j,1} s_{1,i} + a_{j,2} s_{2,i}  + ... + a_{j,d_{x}} s_{d_{x},i}$$

where $$a_{jk}$$ is our mixing matrix.
So each microphone has its own associated constant mixing vector.
Dropping the $$i$$ subscript, we can say
$$\v{x} = \v{A}\v{s}$$
where $$\v{x}\in \mathbb{R}^{d_x}$$ and $$\v{s}\in \mathbb{R}^{d_s}$$ are vectors holding the multitrack recordings and instrument-latents, respectively, at a moment in time, and $$\v{A}$$ is our _mixing matrix_.
Overall we can say that all $$N$$ slices of our recording are the result of a linear mapping of this type:

$$\v{X} = \v{AS} \label{eq:block_gen}$$

If we knew $$v A$$, then the problem would be solved -- we could simply invert $$\v A$$, giving us the unmixing matrix $$\v{A}^{+} = \v{A}^{-1}$$, and our inferred sources would then be $$\v{S}_{\mathrm{inferred}}=\v{A^{-1} X}$$ (if $$d_s < d_x$$, $$\v{A^{-1}}$$ is the [psuedo-inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)).

So, to proceed with unmixing our signal, trying to get some values for $$\v{s}$$ given our recorded values of $$\v{x}$$ we are going to have to find $$\v{A}$$.
There are a whole bunch of ways to proceed further within this problem within the ICA framework.

Part of what makes ICA somewhat confusing to study is that one can be solving essentially the same problem, inverting a mixing process like this using very different looking mathematical tools.

The most obvious approach perhaps is to specify that we wish to learn a mapping, an unmixing, that results in a maximumly-independent latent representation of our data -- a multivariate random variable is statistically independent if its probability distribution is equal to the product of its marginals.
As discussed in my post on [VAE robustness](https://matthewwilletts.github.io/Defending-VAEs-from-Adversarial-Attack/), an intuitive way to measure how close-to-mutually-independent a multivariate distribution would be to see how well approximated it is by a product of its marginal distributions under some objective.
One way to captured this is the _total correlation_ (Watanabe, 1960), the $$\KL$$ between these two entities:
$$ \mathrm{TC}(p(\v{a})) = \KL\left(p(\v a) \mid\mid \prod_{j=1}^{d} p(a_j) \right)$$
where $$d$$ is the dimensionality of the variable at hand.
Thus the Total Correlation is one such measure of how close to being statistically independent a multivariate probability distributions is.
If we try to train a mapping into the latent space from data space that minimises this quantity over our dataset, we would be doing ICA (Everson & Roberts, 2001).

This approach can work very well.
A powerful approach that is based on that approach is FastICA.

|  Blind Source Separation using FastICA |
|:-:|
|![BSS](/images/ica/sphx_glr_plot_ica_blind_source_separation_001.png#center){:height='400px'}
|Here $$d_x=d_s=3$$, so we have recordings by three 'microphones' (top panel) that are linearly-mixed versions of three 'instruments' (middle panel) which are the latents we wish to infer. ICA unmixing (third panel) does achieve this, while PCA analysis (bottom panel) does not. Note that the ICA-output unmixing is only accurate up to scalings and flips of the true sources -- all are of shrunked magnitude, and the sawtooth component has been flipped.|
|Reproduced from [_scikit-learn_](https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html)|

But, despite its success, that is not quite the way we will proceed.
Back in the 1990s, the papers MacKay (1996) and Cardoso (1997) showed that fitting an ICA model that aims to maximise the total correlation is equivalent to fitting and performing inference in a 'matched' generative model.
This method naturally allows us to extend our approach: we can choose (up to certain requirements) the prior for our latents, the generative process can involve noise (i.e. measurement error) and we can choose from a variety of methods of inference.

## Probabilistic Linear Independent Components Analysis

Here we will take a probabilistic view for linear ICA, specifiying a probabilistic model for our data $\v x$ with parameters we will try to learn, and trying to perform inference on the unknown latent variables $\v s$. 
Within that model we will try to find an appropriate unmixing matrix.

So, here we need to specify a probabilistic generative model that contains linear mixing of latents to produce data, $$\v{x} = \v{As}$$, at its core, and one that imposes statistical independence between latents.
First, we will need a prior over the latents, the sources: $$p(\v{s})$$.
We want each latent to be 'about' a single aspect of the data.
I.e. we are asking for our representations to be in some sense _axis-aligned_.
Perhaps surprisingly, placing separate independent, zero mean, unit variance Gaussian priors over the latent dimensions is not acceptable for our purposes -- but why this does not work can be seen immediately if we think geometrically.

|  Spheres rotate into other sphere, so priors with spherical contours are not appropriate for ICA|
|:-:|
|![A sphere](/images/ica/Euler_AxisAngle.png#center){:height='300px'}
|The contours of a product of Gaussians are spheres. Spheres rotate into other spheres, so we can place our axes down arbitrarily and obtain the same distribution. So if our aim is to learn axis-aligned latents, this is bad choice.|
|Reproduced from [_wikimedia_](https://commons.wikimedia.org/wiki/File:Euler_AxisAngle.png)|

Imagine that we obtain a model where our latents are drawn from this prior.
If we apply an arbitrary rotation in the latent space we will have a new model that is indistinguishable from the initial one in terms of its statistical properties, but we have achieved this new model by mixing-together our latents.
This doesn't sound promising if we ant to learn an unmixing our our data in our latent space, if arbitrary re-mixing in the latent space is a baked-in property!

Put another way, a rotation of a product of standard Gaussians is itself a product of standard Gaussians.
If we are interested in learning independent latent representations to undo a linear mixing process, Eq $$\eqref{eq:block_gen}$$, then if that model is indistinguishable up to a whole class of linear transformations, the approach is not going to work.
This is why the PCA results, in the lowest panel of the plot up above, are not giving us what we want.

So, if we are going to work in a generative modelling-framework, we need a generative model that breaks this symmetry.
It turns out that pretty much any standard distribution that isn't a Gaussian works for our purposes -- it can be either heavier-tailed than a Gaussian or lighter-tailed.
That then gives us a axis-aligned contours in the prior in the latent space.
A product of uniform distributions is a common choice, as is a Laplace distribution.
We can parameterise a whole family of ICA-appropriate prior distributions using the Generalised Gaussian (GG)

$$p(\v{s})=\prod_{i=1}^{d_s} \mathrm{GG}(s_i\mid\mu,\alpha,\rho) , \qquad \mathrm{GG}(s_i|\mu,\alpha,\rho) =\frac{\rho}{2\alpha \Gamma(1/\rho)}\exp\left[{\left(-\frac{\lvert s_i - \mu\rvert}{\alpha}\right)^\rho} \right]$$

with mean $$\mu$$, scale $$\alpha$$ and shape $$\rho$$.
For $$\rho=2$$ we recover a Gaussian distribution (so we cannot have $$\rho=2$$), and for $$\rho=1$$ we have the (heavy-tailed) Laplace. 
As $$\rho \to \infty$$ the distribution becomes increasingly sub-Gaussian, tending to a uniform distribution.
(In the paper, we use $$\rho=10$$ in our experiments where we want something approximately uniform, and $$\rho=1$$ when we want something heavy tailed.)

Again, as discussed in my post on [VAE robustness](https://matthewwilletts.github.io/Defending-VAEs-from-Adversarial-Attack/), we can use _variational inference_ to parameterise _posterior distributions_ over the latent sources in our model.
Following very much a VAE-like approach, we introduce an amortised posterior for $$\v s$$:

$$q_\phi(\v s \mid \v x) = \mathcal{N}(\v{\mu}=\v{A}^{-1}\v{x}, \v \sigma = \v\sigma_\phi)$$

And we will say our data was generated via a likelihood $$p_\theta(\v{x}\mid\v{s})$$ of the location-scale family with location $$=\v{A}\v{s}$$ and fixed scale.

Training our generative model and variational posterior then means tuning $$\v A$$ [including via its (pseudo-)inverse] and $$\sigma_\phi$$ to maximise a lower bound on the overall log-probability of our data under out model, the ELBO:

$$\ELBO(\v{X};\v{A},\sigma_\phi) = \expect_{\v{x}\sim\v{X}}\left[\expect_{\v{s}\sim_\phi(\v s \mid \v x)}\left[\log p_\theta(\v{x}\mid\v{s})\right]- \KL\left(q_\phi(\v s \mid \v x)\mid\mid p(\v{s}\right) \right].$$

|  Linear Unmixing using Variational Inference |
|:-:|
|![Unmixing](/images/ica/linear_ICA_dsp.png#center){:height='150px'}
|Linear Unmixing using this Variational Inference approach. We take two dSprites images (a) and generate our dataset by making numerous linear mixings of them (b). Our approach (d) learns the true underlying images (the entries in $$\v A$$) much in the same way as FastICA (c) -- for full details of our implementation, see the paper.|

This is a powerful method, and works well, as long as the problem conforms to the very strict assumptions we have made about how the data was generated.
While the linear-mixing assumption is all well and good for things like audio signals, what if we are interested in high dimensional data, say images of faces like we discussed right at the start?
The idea of **linearly** mixing _smiling_ or _age_ directly to the pixel values of the images is a bit of a non-starter.
So what if we are interested in non-linear generative models for our data, and therefore have to consider _non-linear unmixings_ to discover the latent sources?

## Flows and Non-linear ICA

The first problem with extending ICA to non-linear mixing is the problem of inverting a non-linear operation.
In the above background section, we skated over the difficulties of having the dimensionality of the data $$d_x$$ and the dimensionality of the sources $$d_s$$ vary.
In the linear setting the problem is pretty easy to handle if $$d_s\leq d_x$$.
In the non-linear case, however, if we want to maintain invertibility, if we want the core mapping at the heart of our generative model, $$\v x = f(\v s)$$, to be an bijection between $$\mathcal{X}=\mathbb{R}^{d_x}$$ and $$\mathcal{S}=\mathbb{R}^{d_s}$$, then really we have to have $$d_x=d_s$$.

ICA models, linear or not, with $$d_x=d_s$$ are called _square_, in analogy with the linear case for which this design choice means the linear mixing and unmixing operations are performed by square matrices.
Early work in the 1990s by Deco & Brauer (1995) and Parras et al. (1995) proposed models where the non-linear function $$f(\cdot)$$ is learnable from the data and _is invertible by construction_.
This means that we can then use $$f^{-1}(\v x)$$ as (part of) the method of inference in giving us our latent representations of the sources.

These ideas have come back to the fore in modern machine learning under the name _normalising flows_.
Here we transform between probability distributions using an **invertible** function via the change-of-variables formula.
If we have a variable $$\v z\in\mathcal{Z}=\mathbb{R}^{d_x}$$ with simple base distribtion $$p(\v z)$$, we can specify a distribution over data $$\v x\in\mathcal{X}=\mathbb{R}^{d_x}$$ as

$$p(\v x) = p(\v z) \left\lvert \mathrm{det} \frac{\partial f^{-1}}{\partial \v z}\right\rvert
\label{eq:changeofvar}$$

where $$f$$ is a bijection from $$\mathcal{Z}\rightarrow\mathcal{X}$$.
In essence under this model the generative process is that our data was made by sampling a latent variable $$\v z$$ and mapping it through a function $$f$$, and as the function is invertible by construction we can perform inference by simply applying $$f^{-1}$$ to our data.
It is essential that that $$f$$ be richly-parameretised, so give it the necessary power and flexibility to map between latents and data.
Clearly this is a generalisation of square ICA to non-linear mixings.

If $$f$$ has a tractable Jacobian we can train this model by pure maximum likelihood on our data.
The Jacobian being lower-traingular is a common choice -- the determinant is then simply the product of the diagonal elements.
In essense, we transform our data to representations in $$\mathcal{Z}$$ and tune the parameters of $$f$$ to maximise $$\log p(\v x)$$ via Eq$$~\eqref{eq:changeofvar}$$.

In specifying these flow models it is standard to compose them as a pipeline of simpler invertible functions that are composed together and in doing so retain the simple structure of their Jacobian that renders training tractable.

For more flexible distributions for $$\v{x}$$, we can use function composition to 
specify $$\v{x}$$ through a series of composed functions, from our simple initial $$p$$ into a more complex multi-modal distribution; for example for a series of $$K+1$$ mappings,

$$\v{z} = f_K \circ ... \circ f_0(\v{x}).$$

By the properties of determinants under function composition 
$$\begin{equation}
p(\v{x}) = p(\v{z}_K)\prod_{i=0}^{K}\left\lvert\mathrm{det}\frac{\partial f_{i}^{-1}}{\partial \v{z}_{i+1}}\right\rvert,
\label{eq:deepflow}
\end{equation}$$
where $$\v{z_i}$$ is the variable resulting from the transformation $$f_{i}(\v{z}_{i})$$, $$p(\v{z}_K)$$ defines a density on the $$K^{\mathrm{th}}$$, and the bottom most variable is our data ($$\v{z}_0 = \v{x}$$).

The first paper to bring these ideas back to the mainstream in modern ML was NICE: Non-linear Independent Components Estimation (Dinh et al., 2015).
The exact method used in that paper for building the invertible 'building blocks' of the model has now been superceded -- check the paper out though if you are interested.
For us a key take-away, though, is that this is a _variety_ of non-linear ICA; the hint is in the name.
Is not, however, quite what we want.

The first thing to fix is that NICE (and subseqent models with richer invertible 'building blocks') uses $$\mathcal{N}(\v 0,\v 1)$$ as the base distributions for $$\v z$$.
As discussed in the background section on Linear ICA above, Gaussians are not suitable for ICA models as they are unchanged by rotations, so intrinsically they cannot learn the axis-aligned representations we wish to obtains.
Secondly, this method is intrinsically dimensionality preserving, and really when considering high-dimensional data like images we want to learn a latent representation of a smaller dimensionality -- we don't think that as many latents are there are sub-pixels are needed to model an image.

So, how can we try to make a model that uses the powerful bijective functions of modern flows, including more expressive consituent functions that early methods like NICE, but also give us some compression: learning a reduced number of ICA-latents?

## Bijecta: Combining Flows with Linear ICA

In our paper we use a modern form of flow, a Rational Quadratic Spline (RQS) flow (Durkan et al., 2019), not as the full model but as a sub-component, as a _feature extractor_ that outputs a representation $$\v z\in\mathbb{R}^{d_x}$$.
That representation is now an intermediate value that we _then_ compress using a non-square linear ICA model.
The idea is that we want to learn a mapping between data $$\v x$$ and representations $$\v z$$ for which non-square linear-ICA is an easy model to learn.
It is as if we are _learning the data_ for the linear ICA model.

|  Plate Diagrame for Bijecta |
|:-:|
|![Bijecta](/images/ica/bijecta.png#center){:height='150px'}
|[Left] Generative model and [Right] Approximate Posterior for Bijecta|

So here we have a generative model where

$$\begin{align}
    &p(s_i) = \mathrm{GG}(s_i |\mu=0,\alpha=1,\rho), {\mathrm{for\ } i \in \{1,\dots,d_s}\} \\
    &p(\v{z}|\v{s}) = \mathcal{N}(\v{x}|\v{A}\v{s}, \v{\Sigma}_\theta) \\
    &p_\theta(\v{x},\v{s})=p_\theta(\v{x}|\v{s})p(\v{s})=p(\v{z}|\v{s})p(\v{s})\left\lvert\mathrm{det}\frac{\partial f_\theta^{-1}}{\partial \v{z}}\right\rvert
\end{align}$$

where $$\v{A} \in \mathbb{R}^{d_x\times d_s}$$ is our (unknown) ICA mixing matrix, which acts on the sources to produce a linear mixture, expanding the dimensionality to $$d_x > d_s$$ in the process; and 
$$\v{\Sigma}_\theta$$ is a learnt or fixed diagonal covariance.
This linear mixing of sources yields an intermediate representation $$\v{z}$$ that is then mapped to the data by a flow. 
Our model has three sets of variables: the observed data $$\v{x}$$, the flow representation $$\v{z} = f^{-1}(\v{x})$$, and ICA latent sources $$\v{s}$$.


We choose a linear mapping in our posterior, with $$q_\phi(\v{s}|\v{z}) = \mathrm{Laplace}(\v{s}|\v{A}^{+}\v{z}, \v{b}_\phi)$$,
where we have introduced variational parameters $$\phi=\{\v{A}^{+}, \v{b}_\phi\}$$ corresponding to an unmixing matrix and a diagonal diversity.

Roughly speaking, we are saying that our learnt latent sources are

$$\v s = \v{A}^+ f^{-1}_\theta(\v x) + \v b_\phi\circ\v \eta,$$

where $$\v{A}^+ \in \mathbb{R}^{d_s\times d_x}$$ acts to 'unmix' the flow outputs and compress them to dimensionality $$d_s < d_x$$ in the process.
$$\v\eta\sim \mathrm{Laplace}(\v 0,\v 1)$$ gives us the noise associated with the amortised posterior $$q_\phi$$.


Using samples from this posterior we can define a lower bound $$\ELBO$$ on the evidence in $$\mathcal{Z}$$

$$\begin{align}
\log p(\v{z};\v{A},&\v{\Sigma}_\theta)  \geq \ELBO(\v{z};\phi,\v{A},\v{\Sigma}_\theta) = \expect_{\v{s}\sim q}[\log p(\v{z}|\v{s}) -\KL(q_\phi(\v{s}|\v{z})|| p(\v{s})).
\label{eq:elbo_z_ica}
\end{align}$$

Using the change of variables equation, Eq $$\eqref{eq:changeofvar}$$, and the lower bound on the evidence for ICA in $$\eqref{eq:elbo_z_ica}$$ for $$\mathcal{Z}$$, we can obtain a variational lower bound on the evidence for our data $$\v{x}$$ as the sum of the ICA model's ELBO (acting on $$\v{z}$$) and the log determinant of the flow:

$$\begin{align}
\log p_\theta(\v{x};\v{A}, \v{\Sigma}_\theta)  \geq  \ELBO(\v{x};\theta, \phi,\v{A}, \v{\Sigma}_\theta) = \ELBO(\v{z};\phi,\v{A},\v{\Sigma}_\theta)
+ \log \left\lvert\mathrm{det}\frac{\partial f_\theta^{-1}}{\partial \v{z}}\right\rvert
\label{eq:total_var_objective}
\end{align}$$

As such our model is akin to a flow model, but with an additional latent variable $$\v{s}$$; the base distribution $$p(\v{z})$$ of the flow is defined through marginalizing out the linear mixing of the sources.

For this method to work you might think you have to train the flow part of the model and the linear-ICA model that sits atop it at the same time.
And, to an extent, that is true -- we want the flow to learn to output good $$\v z$$ representations that are easy for the linear-ICA model to act on.

There are problems, however, with training two models on top of each other -- for example with GANs it is well known the discriminator is often much faster to train than the generator, so some hyperoptimisation or grid search is often needed to find some learning procedure that provides both stable training and good models.
Similarly, here we are training a large, powerful flow models that has millions of parameters, alongside a relatively weak linear ICA model.
We want training to be stable and effective, and we want to avoid adding extra design choices like 'having a different learning rate for the linear ICA part of the model' or 'doing $$n$$ update steps of the flow for each single update of the linear ICA model', or whatever.

Recall also that really we are aiming more for the flow to be bent to the will of the linear ICA model: we want the flow to do the heavy lifting to make the linear ICA model's job easy.

To that end, we **fix** the unmixing matrix in the linear ICA model.
We contruct it at init, and it is then fixed.
There is no training to be done for the unmixing matrix, so no tweaking of learning rates or anything.
So not only is the flow trying to output representations that are good for linear ICA, the flow is having to output representatiosn that are good for a particular, fixed unmixing procedure.

(We still learn the mixing matrix.
A naive pseudo-inverse doesn't take into account the geometry of the space $$\mathcal{Z}$$: some directions in $$\mathcal{Z}$$ are more important to get right than others when compressing and re-expanding via $$\mathcal{S}$$, and by learning $$\v A$$ we can take that into account.
In the end the learnt $\v A \approx \left(\v{A}^+\right)^{-1}, but not exactly.)

This then leads to the natural question, what is the right kind of unmixing matrix we should have for this procedure to work well?

For that, we have to return to the world of linear ICA, and consider the manifold of optimal ICA unmixing matrices.

## The Manifold of ICA Unmixing Matrices

With a linear ICA model, recall that $$\v{A}^{+}$$ linearly maps from the data-space $$\mathcal{X}$$ to the source space $$\mathcal{S}$$.
It can be decomposed into three linear operations.
First we _whiten_ the data such that each component has unit variance and these components are mutually uncorrelated.
In the compressive case, where $$d_s < d_x$$, whitening is the step where
We then apply an orthogonal transformation and finally a scaling operation (Hyvarinen et al., 2001, Section 6.34) to 'rotate' the whitened data into a set of coordinates where the sources are independent _and_ decorrelated, and appropriately scaled for the geometry of the sources' priors.
Whitening on its own is not sufficient for ICA -- having no correlation between sources' aggregate posteriors is not the same as those sources being statistically independent.
Put another way, two sources can be uncorrelated \textit{and} dependent.

So we can write the linear ICA unixing matrix as a series of distinct linear operations 

$$\begin{equation}
\v{A}^+ = \v{\Phi R}\v{W}
\label{eq:classic_ica_unmix}
\end{equation}$$

where $$\v{W}\in\mathbb{R}^{d_s\times d_x}$$ is our whitening matrix, $$\v{R}\in\mathbb{R}^{d_s\times d_s}$$ is an orthogonal matrix and $$\v{\Phi}\in\mathbb{R}^{d_s}$$ is a diagonal matrix.

We can show what these parts of the operation do in this figure.

|  Linear ICA Unmixing, Visualised|
|:-:|
|![Linear ICA Unmixing](/images/ica/ICASeq.png#center){:height='220px'}
|The Unmixing process of Eq \eqref{eq:classic_ica_unmix} visualised for a 2D dataset.|

Matrices that can be factored this way are known (slightly confusingly) as _decorrelating matrices_.
Applying a matrix of this type to data, first $$\v W$$ decorrelates, then $$\v R$$ rotates the decorrelated intermediate outputs to give statistical independence, and then $$\v \Phi$$ scales the outputs to fit well the geometry of the prior in $$\mathcal S$$.
The optimum unmixing matrices in Linear ICA are in this family, given enough data (Everson & Roberts, 1999).
We would like to leveage these ideas in thinking about how we can be best choose our unmixing matrix within our Bijecta model.
In overall effect, our combination of a flow and a linear operation has to achieve philosophically similar results to the purely-linear case, but 1) do so while handling non-linearly-mixed data and 2) while actually aiming to have the linear unmixing be _fixed_.

Our aim to merely whiten in the unmixing matrix, and off-load the work that, in the linear case, is done by the square matrix $$\v R$$ and scaling $$\v \Phi$$, to the powerful flow in our model -- but doing these jobs ahead of the whitening operation.
The flow will learn to give representation that are a simple whitening operation from being unmixing -- with statistical independence rewarded by the $$\KL$$ term in the objective Eq $$\eqref{eq:total_var_objective}$$.

Again, we can give a visual representation of our aim here.

|  Bijecta Unmixing, Visualised|
|:-:|
|![Bijecta Unmixing](/images/ica/BIJSeq.png#center){:height='220px'}
|The Unmixing process of Bijecta visualised for 2D a dataset.|


Now, in Linear ICA we construct the whitening matrix $$\v W$$ using Singular Value Decomposition LINK.
This won't suit us -- after all we want our method to be data-agnostic so we can construct $$\v{A}^+$$ 'blind' with no data dependency.

(Of course in the linear case you just have to do this once, just on your observed dataset, so doing SVD isn't that big a deal.
For us, however, we are acting on the output of the flow.
So the equivalent of the 'data matrix' from the perspective of our linear model is our whole data set fed through the current instance of the flow [$$\v Z=f^{-1}_\theta(\v X)$$] which changes after every gradient update to $$\theta$$.
So if we were to embark on an SVD-whitening approach here, we would have to perform SVD after every update, and backprogagate through the procedure as well.
The instability and overhead associated with this are exactly the kind of thing we are trying to avoid -- hence having a fixed unmixing matrix.)

There is a neat way to achieve approximate whitening in a way that doesn't depend on the data at all.
This is possible using _sketching_, also known (more informatively) as 'random projections'.
Remarkably, if we sample the elements of a matrix iid from certain (simple) univariate distributions, then the resuling matrix is an approximate whitening matrix -- see the full paper for theory on this.

A particularly simple sketching matrix is that of Achlioptas (2003), where simply each entry is $$\pm1$$ with equal probability -- and multiplied by an overall scaling to preserve distances in the projected space.

This completes our definition of _Bijecta_ -- now we get to set it loose on some datasets and see how it performs.

## Bijecta Experiments

First, we can check how well this does on a problem that breaks Linear ICA.
We will stick with dSprites, but instead of having the latent sources modulate the _intensity_ of sprites we can use 2 latents to modulate the $$x, y$$ positions of the sprites in the field of view.
The position of the latent is learnt by sampling $$x$$ and $$y$$ each uniformly over the image and then inserting the sprite -- so $$x,y$$ position, or a rotation thereof, are the true, underlying latent variables that explain this data.

And of course this is fundamentally an _affine_ mixing procedure, not linear -- so linear ICA is not an approprate model.

Both the VAE and Bijecta are trained with $$\v z \in \mathbb{R}^2$$, both with Laplace priors (ie priors appropriate for performing ICA).

|  Affine Unmixing  |
|:-:|
|![Unmixing](/images/ica/bij_dsp.png#center){:height='200px'}
|Affine Unmixing using Bijecta. We successfully 'disentangle' the $$x,y$$ positions of a dSprites sprite in an image, obtaining the true underlying latents. Modulating the learnt latent variables leads to clean and consistent changes in the displacement of the sprite. Note that a VAE with an ICA-appropriate prior doesn't learn the true latents. For full details of our implementation see paper.

Clearly Bijecta has learnt the appropriate representation -- traversing the values of $$\v z$$ produces neat, consistent, smooth changes in the placement of the generated sprite -- unlike the VAE.

We can also apply this model to standard machine learning datasets.
We apply this to CelebA, and benchmark again against 
We might hope that 'disentangling' methods for VAEs might work here.
So in addition to a VAE with Laplace priors, we also train $$\beta$$-TCVAEs -- duing training the vanilla ELBO has added to it an upweighted total correlation term for the aggregate posterior.

We measure the total correlation of the resulting aggregate posteriors -- after all mutual independence between latent representations is in some sense a necessary requirement for a successful non-linear ICA algorithm.
All models were trained with 32 dimensional latent variables, and all VAEs use the
same architecture and training as Chen et al. (2018).
(This test is also particularly tilted towards rewarding our $$\beta$$-TCVAE baseline, which after all _directly penalises exactly this quantity during training_.)

We find that Bijecta achieves much lower TC values

|:-:|:-:|:-:|:-:|
| |Laplace-VAE|$$\beta$$-TCVAE|Bijecta|
|_**TC**_:|106.7 ± 0.9 | 55.7 ± 0.1 | **13.1 ± 0.4**|

Here we measure the
source separation of different models on CelebA:
the TC of the validation set embeddings in the latent space of: Laplace prior VAEs, β-TCVAEs
(β = 15), and Bijecta with a Laplace prior (± indicates
the standard deviation over 2 runs).

This is all pretty nice.
As a final test, we try to measure how much the Bijecta training procedure compresses information into a low dimensional subspace, reading to then be extracted by $$\v A^+$$, compared to training the flow model with a standard base distribution.
We can measure this by how much of the variance in $$\mathcal{Z}$$ we can explain as we build up an orthogonal basis in that space.

|Spectra of $$\mathcal{Z}$$ representations for Flows and Bijecta, on F-MNIST and CIFAR-10|
|:-:|
|![Spectra](/images/ica/spectra.png#center){:height='150px'}
|Explained variance plots for the embedding in $$\mathcal{Z}$$, as measured by the sums of the eigenvalues of the covariance matrix of the embeddings, for both our Bijecta model and for an RQS model of equivalent size trained with a Laplace base distribution. For both Fashion-MNIST (left) and CIFAR 10 (right) datasets we see that the Bijecta model has learned a compressive flow, where most of the variance can be explained by only a few linear projections. The shaded region denotes the first 64 dimensions, corresponding to the size of the target source embedding $$\mathcal{S}$$.

This shows that Bijecta is computing representations that are easy to compress with a simple non-square whitening transform without loosing much in the process, unlike vanilla RQS flows

Overall, we think this is a fun approach, pairing powerful flows with (partially) fixed linear models.
[Check out the full paper](https://arxiv.org/abs/2002.07766) for more experiments and for the more technical aspects of the idea.

[^1]: Admittedly this is an appoximation -- maybe _wearing sunglasses_, being _young_ and _smiling_ are in fact interdependent, but hopefully the correspondance is weak. If wearing sunglasses and smiling, say, really did strongly tend to go together in some way, then a model of this type _would_ learn to lump them together as a single _wearing sunglasses while smiling_ variable.


#### References

**Dimitris Achlioptas**. _Database-friendly random projections: Johnson-Lindenstrauss with binary coins_. In Journal of Computer and System Sciences, volume 66 (pp. 671–687), 2003.

**Jean-Francois Cardoso**. _Infomax and Maximum Likelihood for Blind Source Separation_. IEEE Letters on Signal Processing, 4, 112–114, 1997.

**Ricky T Q Chen, Xuechen Li, Roger Grosse, and David Duvenaud**. _Isolating Sources of Disentanglement in Variational Autoencoders_. (arXiv:1802.04942) In NeurIPS, 2018.

**Gustavo Deco, Wilfried Brauer**. Higher Order Statistical Decorrelation without Information Loss. In NeurIPS, 1994.

**Laurent Dinh, David Krueger, and Yoshua Bengio**. _NICE: Non-linear Independent Components Estimation_. In ICLR, 2015.

**Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios**. _Neural Spline Flows_. In NeurIPS, 2019.

**Richard Everson & Stephen J Roberts**. _Independent Component Analysis: A Flexible Nonlinearity and Decorrelating Manifold Approach_. Neural Computation, 11(8), 1957–83, 1999.

**Richard Everson & Stephen J Roberts**. _Independent Component Analysis_. Cambridge University Press., 2001.

**Aapo Hyvärinen, Juha Karhunen, and Erkki Oja**. _Independent Component Analysis_. John Wiley, 2001.

**Diederik P Kingma and Max Welling**. _Auto-encoding Variational Bayes_. (arXiv:1312.6114) In ICLR, 2014

**Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf and Olivier Bachem**. _Challenging common assumptions in the unsupervised learning of disentangled representations_. (arXiv:1811.12359) in ICML, 2019

**David J C Mackay**. _Maximum Likelihood and Covariant Algorithms for Independent Component Analysis_. Technical report, University of Cambridge, 1996.

**Lucas Parra,Gustavo Deco, and Stefan Miesbach**. _Redundancy reduction with information-preserving nonlinear maps_. Network: Computation in Neural Systems, 6(1), 61–72. 1995.

**Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra**. _Stochastic Backpropagation and Approximate Inference in Deep Generative Models_. (arXiv:1401.4082) In ICML, 2014

**Michal Rolinek, Dominik Zietlow,and Georg Martius**. _Variational Autoencoders Pursue PCA Directions (by Accident)_. (arXiv:1812.06775) In CVPR, 2019

**Satosi Watanabe**. _Information Theoretical Analysis of Multivariate Correlation_. IBM Journal of Research and Development, 4(1):66–82, 1960