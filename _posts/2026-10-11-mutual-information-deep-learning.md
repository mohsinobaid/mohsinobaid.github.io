---
layout: post
title: "Squeezing More Information Out of Deep Learning"
subtitle: "What I learned bolting mutual information estimators onto VAEs and SimCLR"
date: 2026-10-11
author: "Moss"
header-img: "img/bg-little-universe.jpg"
mathjax: true
tags: [Machine Learning, Deep Learning, Information Theory]
comments: true
---

Back in 2021 I finished my MEng at Ryerson (now Toronto Metropolitan University) with a dissertation called *Information Theoretic Measures applied to Deep Learning Models*. It was supervised by Dr. Nariman Farsad and Dr. Sharareh Taghipour, and a lot of it took shape in the weekly meetings of Dr. Farsad's Learning and Inference lab.

If you've read my [divergence measures post]({% post_url 2021-05-25-Divergence-Measures-Breakdown %}), you've seen the very beginning of that rabbit hole. This post covers the rest of it: what I was trying to do, what I ran, and what actually happened. I've also tried to be honest about the parts that didn't work.

## The question

Labelled data is expensive and unlabelled data is everywhere, so a lot of the interesting work in deep learning goes into **unsupervised representation learning**. You train a network on raw data and hope that its internal "latent space" ends up organising the data in a useful way, for example with all the cats near each other and away from the dogs.

The usual way to check is a **linear probe**. Freeze the trained network, train a single linear layer on top of its latent vectors, and see how well it classifies. If a straight line can separate the classes, the representation has untangled something real.

My question was:

> If you explicitly tell a network to maximise the **mutual information** between its input and its representation, do you get a better representation?

I tried this on two very different architectures: **Variational Autoencoders** and **contrastive learning (SimCLR)**.

## A 60-second mutual information primer

Mutual information (MI) measures how much knowing one variable tells you about another:

$$I(X;Z) = H(X) - H(X \mid Z)$$

You can also write it as a KL divergence between the joint distribution and the product of the marginals:

$$I(X;Z) = D_{KL}\big(p(x,z) \,\Vert\, p(x)\,p(z)\big)$$

If $$X$$ and $$Z$$ are independent, the two distributions are identical and MI is zero. The more $$Z$$ "remembers" about $$X$$, the bigger the gap.

The catch is that you can't compute this exactly for images and 512-dimensional embeddings. Instead you train a small neural network (a "critic" $$T$$) to *lower-bound* it. I used three estimators:

**MINE** ([Belghazi et al., 2018](https://arxiv.org/abs/1801.04062)) uses the Donsker–Varadhan representation of KL:

$$I_{MINE} = \mathbb{E}_P[T] - \log \mathbb{E}_Q[e^{T}]$$

It's flexible, but the $$\log \mathbb{E}[e^T]$$ term makes it high variance, so it can be noisy.

**SMILE** ([Song & Ermon, 2020](https://arxiv.org/abs/1910.06222)) is the same idea with the exponential term clipped to $$[e^{-\tau}, e^{\tau}]$$. Giving up a bit of bias buys a big drop in variance:

$$I_{SMILE} = \mathbb{E}_P[T] - \log \mathbb{E}_Q\big[\text{clip}(e^{T}, e^{-\tau}, e^{\tau})\big]$$

**InfoNCE / CPC** ([van den Oord et al., 2018](https://arxiv.org/abs/1807.03748)) is the one hiding inside most contrastive losses. It has low variance but a hard ceiling: **it can never estimate more than $$\log(\text{batch size})$$**. With a batch of 256 that's about 5.5 nats, no matter how much information is really there.

That last point ended up driving the second half of the project.

## Experiment 1: MI as a regulariser in VAEs

A VAE learns by maximising the ELBO, a reconstruction term minus a KL term that pulls the latent code toward a standard normal:

$$\mathcal{L}_{VAE} = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - D_{KL}\big(q(z \mid x) \,\Vert\, p(z)\big)$$

A known failure mode is that the KL term pushes the latent codes into a blurry blob in the middle, so the representation doesn't keep much useful information about the input. **InfoMax-VAE** ([Rezaabad & Vishwanath, 2020](https://arxiv.org/abs/1912.13361)) fixes this by adding an explicit MI bonus, estimated by a separate critic network:

$$\mathcal{L}_{InfoMax} = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \beta\, D_{KL}\big(q(z \mid x) \,\Vert\, p(z)\big) + \alpha\, I(x; z)$$

My twist was to swap in **SMILE** as the estimator for that $$I(x;z)$$ term, on the theory that a lower-variance estimate gives a cleaner training signal.

**Setup:**
- 5 conv layers in the encoder, 5 deconv layers in the decoder, 20-dimensional latent space, batch size 100
- MI critic: 3 linear layers with ReLU
- Evaluation: a linear classifier on the latent vector $$z$$

**MNIST**

| Model | Linear probe accuracy | Estimated MI |
|---|---|---|
| VAE (baseline) | **0.9164** | 2.53 |
| InfoMax (MINE) | 0.9069 | 3.50 |
| InfoMax (MINE, non-permute) | 0.9076 | 7.13 |
| InfoMax (SMILE, non-permute) | 0.9108 | **8.71** |

**CIFAR-10**

| Model | Linear probe accuracy | Estimated MI |
|---|---|---|
| VAE (baseline) | 0.3174 | 2.81 |
| InfoMax (MINE) | 0.4019 | 3.67 |
| InfoMax (MINE, non-permute) | 0.4490 | 8.15 |
| InfoMax (SMILE, non-permute) | **0.4528** | **8.65** |

What I take from this:

- **CIFAR-10 is where it pays off.** Linear probe accuracy went from about 32% to about 45%, a 13-point jump, and it tracks the MI estimate almost perfectly: more MI, better probe.
- **MNIST is already easy.** The plain VAE already sits above 91%, and the MI-regularised versions land within a point of it, slightly *below* in fact. When the classes are that easy to separate, there isn't much left for the MI term to untangle.
- **SMILE gave the highest MI estimate on both datasets** and the best accuracy of the MI variants, which fits the bias/variance argument.

For context, a fully supervised CNN gets about 98% on MNIST and about 83% on CIFAR-10. A 45% linear probe on an unsupervised VAE latent is nothing to brag about in absolute terms, but the *relative* improvement is the interesting part.

## Experiment 2: MI in contrastive learning (SimCLR)

[SimCLR](https://arxiv.org/abs/2002.05709) works like this: take an image, make two random augmentations of it (crop, colour jitter, and so on), run both through an encoder, and train so the two views land close together while everything else in the batch is pushed away. The loss (NT-Xent) is basically InfoNCE:

$$\ell_{ij} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

So SimCLR is quietly maximising mutual information between the two views, and it inherits InfoNCE's $$\log(\text{batch size})$$ ceiling. That's part of why SimCLR famously wants huge batches.

My hypothesis was that if I add an MI estimator that *doesn't* have that ceiling (SMILE), the network might get a stronger signal at smaller batch sizes. The combined loss was:

$$\mathcal{L} = (1-\gamma)\cdot \mathcal{L}_{\text{NT-Xent}} + \gamma \cdot \mathcal{L}_{\text{MI}}$$

I used a ResNet-18 encoder, with experiments on CIFAR-10 and STL-10.

**Step 1: SMILE on its own (35 epochs).** First I replaced NT-Xent entirely with SMILE between the augmented views and swept the hyperparameters. Test accuracy landed in a tight band of about 69–71% across learning rates, clipping values ($$\tau$$ = 0.2 / 0.5 / 0.8 / ∞) and batch sizes (128 / 256 / 512). The best run was 71.1%, with a bigger critic network. So SMILE on its own *works*, and it's comparable to a short SimCLR run, but it isn't a breakthrough.

**Step 2: mixing the two (80 epochs, batch 256).** Then I swept $$\gamma$$:

| γ (MI weight) | Test accuracy |
|---|---|
| 0 (pure SimCLR) | 78.98 |
| 0.006 | 79.13 |
| 0.008 | 79.24 |
| **0.01** | **79.47** |
| 0.02 | 79.34 |
| 0.04 | 78.86 |
| 0.1 | 78.50 |
| 0.2 | 78.27 |
| 0.4 | 77.43 |
| 0.6 | 76.44 |
| 0.8 | 76.11 |
| 1 (pure MI) | 74.54 |

There's a clear shape here. A *tiny* amount of MI loss (γ ≈ 0.01) gives a small bump, about half a point. Beyond about 0.03, accuracy slides steadily down until pure MI is about 4.5 points worse than pure SimCLR.

To be honest about it, half a point from single runs is within the range random seeds alone can produce. The clean result is the downward trend: **adding a SMILE term didn't meaningfully improve SimCLR at batch size 256, and leaning on it hurts.**

## So... does maximising MI help?

My overall verdict: **it depends a lot on the architecture and the task.**

1. **In VAEs, yes, clearly on a harder dataset.** The VAE objective doesn't naturally reward keeping information in $$z$$, so an explicit MI term fills a real gap. A lower-variance estimator (SMILE) helps a bit more.
2. **In contrastive learning, not really.** SimCLR is *already* an MI maximiser. Adding a second, differently biased MI estimate on top mostly competes with the first one. The batch-size ceiling on InfoNCE is real in theory, but at batch 256 it wasn't the bottleneck.
3. **A higher MI estimate isn't automatically a better representation.** It lined up nicely in the VAE experiments, but MI is invariant to any invertible transformation. A representation can hold *all* the information and still be badly tangled for a linear classifier. Tschannen et al. make this argument well in [*On Mutual Information Maximization for Representation Learning*](https://arxiv.org/abs/1907.13625), which I wish I'd read earlier.

## What I'd do differently

Looking back with a few more years of experience:

- **Run multiple seeds.** Most of my contrastive results are single runs, which makes sub-1% differences impossible to call.
- **Actually test small batches.** The whole contrastive hypothesis was about *small* batches, but the γ sweep was all done at 256. The interesting experiment is probably γ at batch 32 or 64, where InfoNCE's ceiling (log 32 ≈ 3.5 nats) really bites.
- **Weaken the augmentations.** Augmentation strength largely determines *what* information SimCLR keeps. With weaker augmentations, the MI term might have more room to contribute.
- **Compare against more VAE baselines** (β-VAE, InfoVAE) on the same footing, not just the vanilla VAE.

If any of this is useful to you, or you've run into the same "MI went up, accuracy didn't" puzzle, leave a comment below. I'm happy to dig up details.

*Thanks again to Dr. Farsad, Dr. Taghipour, and everyone in the LIA lab.*
