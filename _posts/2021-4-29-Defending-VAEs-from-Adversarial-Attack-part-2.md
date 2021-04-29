---
layout: post
title: Improving VAEs' Robustness to Adversarial Attacks – Part 2/2
---


This is part two of a two-part series where I explain in some detail the ideas in my upcoming [ICLR 2021 paper](https://arxiv.org/abs/1906.00230) `Improving VAEs' Robustness to Adversarial Attacks', work done with my friends and colleagues at the University of Oxford and The Alan Turing Institute.

If you are not familiar with Variational Autoencoders, I recommend you start with a review article -- both [Doesch, (2016)](https://arxiv.org/abs/1606.05908) and [Kingma & Welling, (2019)](https://arxiv.org/abs/1906.02691) are great.
$$\renewcommand{\vector}[1]{\boldsymbol{\mathbf{#1}}}$$
$$\renewcommand{\v}{\vector}$$
$$\newcommand\vv[1]{\vec{\v{#1}}}$$
$$\newcommand\KL{\mathrm{KL}}$$
$$\newcommand\TC{\mathrm{TC}}$$
$$\newcommand\W{\mathrm{W}}$$
$$\newcommand\ELBO{\mathcal{L}}$$
$$\newcommand\expect{\mathbb{E}}$$
$$\newcommand\argmin{\mathrm{arg \,min}}$$

[Last time](https://matthewwilletts.github.io/Defending-VAEs-from-Adversarial-Attack/) we gave an introduction to the idea of adversarial attacks on deep nets, how those ideas transpose to attacks VAEs via their latent space, and how regularisation methods that tune the noise of latent representations can be used to induce robust models.
We showed that we can use a particular kind of regularisation, Total Correlation ($$\TC$$) regularisation, in single layer VAEs to induce robustness.
Here we penalise the $$\TC$$ of the aggregate posterior---$$\KL\left(q_\phi(\v z;\mathcal{D}) \mid\mid \prod_{j=1}^{d_z} q_\phi(z_j;\mathcal{D})\right)$$--- as was previously proposed by Chen et al. (2018) as a way to achieve 'disentangled' representations.

Upweighting the aggregate posterior's $$\TC$$ comes a cost of model quality and reconstruction quality, albeit less than that in $$\beta$$-VAEs (where we upweight $$\KL\left(q_\phi(\v z\mid \v x) \mid\mid p(\v z))\right)$$).
Can we develop a model that is robust to adversarial attack while further mitigating this trade-off between robustness and sample quality?

In this post we'll describe how we can get even more robust models by applying the same ideas to _hierarchical_ VAEs.

There is a whole library's worth of papers that improve the quality of VAEs by introducing a _hierarchy_ of layers of latent variables.

These gains stem from using more complicated, hierarchical, latent spaces, rather than less noisy encoders.
So, here we show how these two approaches, extending VAEs to hierarchies of latent variables and tuning the noise in VAEs' latent representations, can be combined to give high quality and robust models.
In fact, the hierarchical-and-regularised models we get from this combination, that we call _Seatbelt-VAEs_, are *significantly more robust to adversarial attacks than single-latent-layer models*.

## Hierarchical VAEs

All the models we kicked around in the previous post had one 'layer' of latent variables.
We can get much, much more expressive models when we have a hierarchy of layers of latent variables.
Recently hierarchical VAEs briefly seized the SOTA-crown for probabilistic modelling of images.
(They had it for about 6 months before score-matching models took the top spot. Perhaps hierarchical VAEs will get it back next conference cycle...)

In hierarchical VAEs we pick some factorisation over latent layers, for the forward and for the inference models, so the layers of latents have some dependency structure between them.
While the component distributions we pick are still (in almost all papers) conditionally-Gaussian (like $$q_\phi(\v z \mid \v x)$$ is a Vanilla VAE), if we marginalise out the conditioning latents then each latent's marginal (but still conditioned on $$\v x$$) distribution is a highly-flexible, non-Gaussian distribution.

So instead of the ELBO being

$$\ELBO(\mathcal{D}; \theta,\phi) = \frac{1}{N}\sum_{i=1}^N
\left[\expect_{\v{z}\sim q_\phi(\v z \mid \v{x}_i)}\left[\log p_\theta(\v{x}_i \mid \v z)\right] - \KL \left( q_\phi(\v z \mid \v{x}_i) \mid\mid p(\v z)\right)\right]$$

it is

$$\ELBO(\mathcal{D}; \theta,\phi) = \frac{1}{N}\sum_{i=1}^N
\left[\expect_{\vv{z}\sim q_\phi(\vv{z} \mid \v{x}_i)}\left[\log p_\theta(\v{x}_i \mid \vv{z})\right] - \KL \left( q_\phi(\vv{z} \mid \v{x}_i) \mid\mid p(\vv{z})\right)\right]$$

where $$\vv{z} = \{\v{z}^1,\v{z}^2,...,\v{z}^L\}$$, our set of layers of latents.

In thinking about how we should best structure our hierarchical VAE for robustness to adversarial attack, we have to consider two aspects.

Firstly, what should the factorisation over latents be in the inference and generative models?
That is, we have to pick how $$q_\phi(\vv{z} \mid \v{x}$$, $$p_\theta(\v{x},\vv{z})$$ should factorise.
Within this, we found conditioning the data likelihood on _all_ layers of latents to be the most important factor.
Consider the alternative -- in some early hierarchical VAEs $$p_\theta(\v{x},\vv{z})=p_\theta(\v{x}|\v{z}^1)p(\vv{z})$$.
This choice means that reconstructions only depend on the samples of $$\v{z}^1\sim q_\phi(\v{z}^1\mid\v{x})$$, so a 'shortcut' exists via a single layer of latents.
The rest of the hierarchical model can 'switch off', with the other parts of the inference model matching the generative model exactly, becoming indifferent to the input data and thus
irrelevant to the model's ability to perform.

If the model reduces to, in effect, a single-latent-layer VAE then the idea that we will be able to improve the model by introducing a hierarchy is for the birds -- the model isn't really hierarchical at all.
So, following lots of other papers (Kingma et al., 2016 is the first), we condition the likelihood on all layers of the hierarchy of latent layers.
Our generative model is thus:

$$p_\theta(\v{x},\vv{z})=p_\theta(\v{x}\mid\vv{z})\prod\nolimits_{i=1}^{L-1} p_\theta(\v{z}^i \mid \v{z}^{i+1}) p(\v{z}^L)$$

For the posterior, in the paper we use the simple 'bottom-up' factorisation:

$$q_\phi(\vv{z}\mid\v{x}) = q_\phi(\v{z}^1\mid\v{x}) \prod\nolimits_{i=1}^{L-1} q_\phi(\v{z}^{i+1} \mid \v{z}^{i}, \v{x})$$

_[We also experimented with the factorisation $$q_\phi(\vv{z}\mid\v{x})=q_\phi(\v{z}_L\mid\v{x})\prod_{\ell=1}^{L-1}q_\phi(\v{z}_\ell\mid\v{z}_{>\ell},\v{x})$$ and found performance to be very similar.]_


This leads to the second choice.
We are armed with the insight, discussed in great detail in the [previous post on robustness in VAEs](https://matthewwilletts.github.io/Defending-VAEs-from-Adversarial-Attack/), that by regularising the latent representations of a VAE we can obtain robust model.
How can we regularise hierarchical VAEs to make them more robust?

We showed that in VAEs where the [total correlation](https://en.wikipedia.org/wiki/Total_correlation) of the aggregate posterior was penalised during training are robust to attacks, and that this method is more effective at inducing robustness than simply upweighting the KL term in the ELBO.
So, we apply this regularisation to the top-most latent variable of our hierarchy.

We call these resulting models _Seatbelt-VAEs_, as the extra straps and toggles add safety.
We can represent deep generative models as _plate diagrams_, as we did in my post on [non-linear ICA](https://matthewwilletts.github.io/Non-Linear-ICA-using-flows-and-fixed-linear-models/).
Each circular node is a probabilistic variables.
Arrows indicated dependency.
Here is an $$L=2$$ Seatbelt-VAE:

|  Plate Diagram for Seatbelt-VAEs|
|:-:|
|![Seatbelt](/images/seatbelt/seatbelt_two_latents_v.png#center){:height='300px'}
|[Left] Generative model and [Right] Approximate Posterior for Seatbelt with $$L=2$$ layers.|
|_Hatching indicates a latent variable with $$\TC$$ penalisation._|

Note that when $$L=1$$ the model reduces to a $$\beta$$-TCVAE and when $$\beta=1$$ it becomes a particular variety of hierarchical VAE with various skip connections.


## Adversarial Attacks on Hierarchical VAEs

We will look at three different attack modes of hierarchical.
The first is a straight forward generalisation of the attack we discussed in the [previous post](https://matthewwilletts.github.io/Defending-VAEs-from-Adversarial-Attack/), attacking via the latent space; we want the variational posterior of the attacked (that is, distorted) input to be close to that of the target datapoint.
We measure closeness using the $$\KL$$ divergence [this attack is first proposed in Tabacof et al., (2016) and Gondim-Ribeiro et al., (2018)].

In our hierarchical model we have multiple _layers_ of latent variables, and the decoder takes _all_ of them as input, so for us to be confident on attacking the image well it is reasonable for the adversary to aim to match the latent representations in all layers.

Thus we have as our attack objective the sum of $$\KL$$s over layers:

$$\Delta^{\mathrm{Seatbelt}}_{KL} (\v{x},\v{d},\v{x}^t; \lambda) = \lambda\mid\mid\v{d}\mid\mid_2 +\sum\nolimits_{i=1}^{L} \KL(q_\phi(\v{z}^i\mid\v{x}+\v{d}),q_\phi(\v{z}^i\mid\v{x}^t))$$


_[[Recent work](https://arxiv.org/abs/2103.06701) has experimented with attacking only certain layers of latent variables in very deep hierarchical VAEs (for example, 32 layers of latents), and finding it is possible to match at different levels of the semantics of the image.
These models, though, do not have $$\TC$$-penalised latents, and are far deeper than the models we study in our paper (we go up to 5 or so layers of latents).
Clearly there is more to be learnt about attacks on very deep VAEs.]_

The second attack is a variation on this: we swap-out the $$\KL$$ divergence in the above attack for the [2-Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric).

$$\Delta^{\mathrm{Seatbelt}}_{W_2} (\v{x},\v{d},\v{x}^t; \lambda) = \lambda\mid\mid\v{d}\mid\mid_2 +\sum\nolimits_{i=1}^{L} \W_{2}(q_\phi(\v{z}^i\mid\v{x}+\v{d}),q_\phi(\v{z}^i\mid\v{x}^t))$$.

The third attack is one previously proposed in Kos et al. (2018) that is applicable generically to VAEs, regardless of their probabilistic structure (though we are the first to apply it to hierarchical VAEs).
We mentioned it in passing in the previous post.
Here the adversary aims to directly maximise the ELBO of the target image under the attacked input, ie a slightly odd form of the ELBO for a VAE where the encoder is fed the attacked point and the decoder the target point:

$$\Delta_\mathrm{output}(\v{x},\v{d},\v{x}^t;\lambda) =\expect_{q_\phi(\vv{z}\mid\v{x}+\v{d})}\left[\log(\v{x}^t\mid\vv{z})\right] - \KL(q_\phi(\vv{z}\mid\v{x}+\v{d})\mid\mid p(\vv{z})) + \lambda\mid\mid\v{d}\mid\mid_2$$

Let now try all this out.
First, let's try to do an attack that's really simple.
We have a dSprites heart and we want to rotate it around and we are going to attack using $$\Delta_{\KL}$$, so via the latent space using the $$\KL$$.
How do different models compare?

|VAE|$$\beta$$-TCVAE, $$\beta=2$$|Seatbelt-VAE, $$\beta=1$$|Seatbelt-VAE, $$\beta=2$$|
|:-:|:-:|:-:|:-:|
|![VAE](/images/seatbelt/rotation_attack_plots_dsprites/0_latent_vae_z64_dsprites_1_mlp_attack_plots.png#center){:height='130px'}|![BTCVAE, beta=2](/images/seatbelt/rotation_attack_plots_dsprites/0_latent_vae_z64_dsprites_2_mlp_attack_plots.png#center){:height='130px'}|![Seatbelt, beta=1](/images/seatbelt/rotation_attack_plots_dsprites/0_latent_vae_2_1_top_True_chain_0.125_attack_plots.png#center){:height='130px'}|![Seatbelt, beta=2](/images/seatbelt/rotation_attack_plots_dsprites/0_latent_vae_2_2_top_True_chain_0.125_attack_plots.png#center){:height='130px'}|

|Within each subplot the top left shows the original input, a heart the right way up. The bottom left shows the target, a heart the wrong way up at the same location in the image. The middle top shows the reconstruction we get from the model when we feed it the original input. The top right shows the adversarial input, so $$\v{x} + \v{d}$$. The bottom right shows $$\v{d}$$, scaled to the range [0,1]. The bottom middle is the most important, the reconstruction given by the model when given the attacked input.|

Clearly we can see that the Seatbelt model is the hardest to attack -- the heart remains the right way up in the adversarial reconstruction (ie the mean of $$p_\theta(\v{x}\mid \vv{z}^*)$$ where $$\vv{z}^*\sim q_\phi(\vv{z}\mid\v{x}+\v{d})$$).
[See the paper for more examples](https://arxiv.org/pdf/1906.00230.pdf).

What about quantitative analysis? How to numerically measure the effectiveness of an attack?
We can perform these three different attacks for Seatbelt-VAEs trained on a range of datasets.

One simple way is to measure the value reached of the attack objective.

_[Digression: This actually isn't quite so simple when we are performing latent-space attacks and comparing between models._

_Consider the case that we are comparing the robustness of a model with lots of layers of latents and one with only a single layer of latents, or with much lower-dimensional latents.
The model with more layers of latents or higher-dim latents will naturally have a larger $$\Delta_{\KL}$$ or $$\Delta_{\W_2}$$ value, as more terms are being added to compute it!
Of course even with this objection, comparing between models of the same architecture but just different strengths of regularisation is totally fine.
And, as we saw last time, $$\TC$$ penalisation increases the value of $$\Delta_{\KL}$$ by orders of magnitude anyway, so we don't have to quibble too much over small changes in dimensionality._

_The second, more subtle, potential issue with using latent-attack $$\Delta$$s to measure robustness is that perhaps different regularisation methods change the intrinsic scale of the aggregate posterior.
We saw last time that $$\beta$$-VAEs, for example, tend to shrink the aggregate posterior to be conterminous with the prior.
So if we were experimenting with $$\beta$$-VAEs and measuring latent $$\Delta$$ values, the trends we might see as we increase $$\beta$$ wouldn't necessarily be purely telling us about how much harder it was to match in the latent space.
It would also be telling us about the overall scale of the learnt representations._

_Concretely, if the aggregate posterior is very spread out then the initial and target datapoints could have posterior representations that are really quite far apart, but in a highly-regularised $$\beta$$-VAE all datapoints get posterior representations that are close to the prior , and thus are close to each other.
So, perversely, a $$\beta$$-VAE could appear to become easier and easier to attack as the regularisation increases, as $$\Delta_{\KL}$$, $$\Delta_{\W_2}$$, and other latent-space attack objectives would naturally shrink._

_For us, using $$\TC$$ penalisation, this is not a concern -- we are penalising the_ symmetry _of the aggregate posterior, not imposing a particular scale.]_

The second way to measure the effectiveness of an attack is to measure the log likelihood of the target given samples from the attacked variational posterior:

$${\log p_{\theta}(\v{x}^t\mid\vv{z}^*)},\,\,\vv{z}^*\sim q_\phi(\vv{z}\mid\v{x}+\v{d})$$

For VAEs with Gaussian data likelihoods with fixed isotropic covariance this reduces simple to measuring the $$L_2$$ distance between the reconstruction we get from the VAE and the target, a totally reasonable 'engineering' measurement of attack quality.

So, having described all this set-up, what do we get?

We trained Seatbelt-VAEs for a range of $$\beta$$ values, on the Faces and on the Chairs datasets, and attacked them with these three different attack methods.
Here we show results for $$L=4$$ models -- for more results over different $$L$$ values [check out the paper](https://arxiv.org/pdf/1906.00230.pdf).
Recall that for $$\beta=1$$ our model is just a hierarchical VAE with some skip connections.

||$${\log p_{\theta}(\v{x}^t\mid\vv{z}^*)}$$, Faces |$$\Delta$$, Faces|$${\log p_{\theta}(\v{x}^t\mid\vv{z}^*)}$$, Chairs|$$\Delta$$, Chairs|
|:-:|:-:|:-:|:-:|:-:|
|$$\KL$$ $$\mathrm{attack}$$|![likelihood faces](/images/seatbelt/attack_plots/faces_deep_x+_adv_likelihood_lineplot_latent_atk.png#center){:height='130px'}|![delta faces](/images/seatbelt/attack_plots/faces_adv_loss.png#center){:height='130px'}|![likelihood chairs](/images/seatbelt/attack_plots/chairs_deep_x+_adv_loss_lineplot_latent_atk.png#center){:height='130px'}|![delta chairs](/images/seatbelt/attack_plots/chairs_adv_loss.png#center){:height='130px'}|
|$$\W_2$$ $$\mathrm{attack}$$|![likelihood faces](/images/seatbelt/attack_plots/faces_deep_x+_adv_loss_lineplot_wass_atk.png#center){:height='130px'}|![delta faces](/images/seatbelt/attack_plots/faces_deep_x+_adv_likelihood_lineplot_wass_atk.png#center){:height='130px'}|![likelihood chairs](/images/seatbelt/attack_plots/chairs_deep_x+_adv_likelihood_lineplot_wass_atk.png#center){:height='130px'}|![delta chairs](/images/seatbelt/attack_plots/chairs_deep_x+_adv_loss_lineplot_wass_atk.png#center){:height='130px'}|
|$$\mathrm{Output}$$ $$\mathrm{attack}$$|![likelihood faces](/images/seatbelt/attack_plots/faces_deep_x+_adv_likelihood_lineplot_output_atk.png#center){:height='130px'}|![delta faces](/images/seatbelt/attack_plots/faces_deep_x+_adv_loss_lineplot_output_atk.png#center){:height='130px'}|![likelihood chairs](/images/seatbelt/attack_plots/chairs_deep_x+_adv_likelihood_lineplot_output_atk.png#center){:height='130px'}|![delta chairs](/images/seatbelt/attack_plots/chairs_deep_x+_adv_loss_lineplot_output_atk.png#center){:height='130px'}|

As you can see, both $$\Delta$$ and log likelihood of the target tend to increase as we increase $$\beta$$.
What about the effect on model quality from $$\TC$$ penalisation?

Of course any additional penalisation terms will lead to some degradation in raw model performance, but how does the trade-off effect Seatbelt-VAEs compared to $$L=1$$ $$\beta$$TC-VAEs??
Pleasantly the raw ELBO (ie evaluated with $$\beta=1$$) decreases more slowly as a function of $$\beta$$ for Seatbelt-VAEs than for $$L=1$$ $$\beta$$-TCVAEs.
So our model not only gives robust models, more robust that $$\beta$$-TCVAEs but also these models have a better trade-off between quality and robustness.

|Faces|Chairs|
|:-:|:-:|
|![Faces ELBO](/images/seatbelt/attack_plots/faces_all_elbo.png#center){:height='200px'}|![Chairs ELBO](/images/seatbelt/attack_plots/chairs_all_elbo.png#center){:height='200px'}|

This is all very nice -- hierarchical VAEs with $$\TC$$ regularisation really offer improved robustness across a range of attacks.
Given the increasing importance placed on model robustness -- [and that VAEs are even being used to defend other models (Ghosh et al., 2018)](https://arxiv.org/pdf/1806.00081.pdf) -- we hope that our approaches and discussion are of some help.
But this is all only the start!

## Conclusion

This has been a bit of a monster two-parter on our ICLR paper, but still this is only scratching the surface for VAE robustness.
If you feel that you have an itch for more, check out our work [on the theoretical aspects of VAE robustness](https://alexander-camuto.github.io/The-Adversarial-Robustness-of-VAEs/), or [drop me an email](mailto:mwilletts@turing.ac.uk).

#### References

**Ricky T Q Chen, Xuechen Li, Roger Grosse, and David Duvenaud**. _Isolating Sources of Disentanglement in Variational Autoencoders_. (arXiv:1802.04942) In NeurIPS, 2018.

**Partha Ghosh, Arpan Losalka, and Michael J Black**. _Resisting Adversarial Attacks Using Gaussian Mixture Variational Autoencoders__. (arXiv:1806.00081) In AAAI, 2019

**George Gondim-Ribeiro, Pedro Tabacof, and Eduardo Valle**. _Adversarial Attacks on Variational Autoencoders_. arXiv:1806.04646, 2018

**Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed and Alexander Lerchner**. _$$\beta$$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework_. In ICLR, 2017a

**Diederik P Kingma and Max Welling**. _Auto-encoding Variational Bayes_. (arXiv:1312.6114) In ICLR, 2014

**Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever and Max Welling**. _Improved Variational Inference with Inverse Autoregressive Flow_. (arXiv:1606.04934) In NeurIPS, 2016

**Jernej Kos, Ian Fischer and Dawn Song**. _Adversarial examples for generative models_. (arXiv:1702.06832) In Workshops of the IEEE Symposium on Security and Privacy, 2018

**Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra**. _Stochastic Backpropagation and Approximate Inference in Deep Generative Models_. (arXiv:1401.4082) In ICML, 2014

**Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus**. _Intriguing properties of neural networks_. (arXiv:1312.6199) In ICLR, 2014.

**Pedro Tabacof, Julia Tavares, and Eduardo Valle**. _Adversarial Images for Variational Autoencoders_. (arXiv:1612.00155), In NeurIPS Workshop on Adversarial Training, 2016

**Satosi Watanabe**. _Information Theoretical Analysis of Multivariate Correlation_. IBM Journal of Research and Development, 4(1):66–82, 1960
