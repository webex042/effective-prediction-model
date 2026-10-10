# Generative Modeling: From Autoencoders to Variational Autoencoders & C-VAEs

> **Stage 7 — Generative Models: Density Estimation, Latent Manifolds, ELBO Derivation & Uncertainty Estimation**

---

## 1. The Generative Paradigm

In previous stages, our predictive models were **discriminative**:
- They modeled conditional likelihoods $p(y \mid x)$ (e.g. classification of equipment failure given sensor features) or regressions $\hat{y} = f(x)$.
- Discriminative models have a fatal limitation: they are blind to the underlying distribution of the input $p(x)$. They cannot tell you if an input sample $x$ came from an improbable out-of-distribution regime, nor can they generate synthetic scenarios or quantify aleatoric vs epistemic uncertainty.

**Generative models** learn the joint distribution $p(x)$ or $p(x, y)$:
- They can draw novel, synthetic samples $x_{\text{new}} \sim p(x)$.
- They can compute likelihoods or reconstruction errors to detect anomalies.
- They map high-dimensional complex telemetry into structured, smooth latent manifolds $z \in \mathbb{R}^{d_z}$.

```mermaid
graph LR
    subgraph Discriminative Model
        D_X[Input Features x] --> D_M[Model f_theta] --> D_Y[Target Class / Forecast y]
    end

    subgraph Generative Latent Variable Model
        G_Z[Latent Prior p(z) ~ N(0, I)] --> G_DEC[Decoder p_theta(x | z)] --> G_X[Generated Sample x]
        G_DATA[Real Data x] --> G_ENC[Encoder q_phi(z | x)] --> G_LAT[Latent Posterior z]
    end
```

---

## 2. Deterministic Autoencoders vs. Probabilistic VAEs

### 2.1 Why Standard Autoencoders Fail at Generation
A standard deterministic autoencoder (AE) optimizes an encoder $z = f_\theta(x)$ and decoder $\hat{x} = g_\phi(z)$ using Mean Squared Error:
$$\mathcal{L}_{\text{AE}}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^N \|x_i - g_\phi(f_\theta(x_i))\|_2^2$$

- **The Problem of Latent Space "Holes"**:
  Without probabilistic constraints, the encoder simply maps data points to arbitrary isolated clusters in latent space.
  Between those clusters lie vast uncharted regions ("holes").
  If you sample a random vector $z \sim \mathcal{N}(0, I)$ and decode it, $g_\phi(z)$ outputs nonsensical, distorted waveforms.
  Standard AEs are great for compression and deterministic reconstruction, but cannot generate.

---

### 2.2 The Variational Autoencoder (VAE) Solution
Kingma & Welling (2013) and Rezende et al. (2014) formulated the VAE as a directed probabilistic latent variable model:

1. **Prior Distribution**:
   $$p(z) = \mathcal{N}(0, I)$$
2. **Generative Model (Decoder)**:
   $$p_\theta(x \mid z) = \mathcal{N}(g_\mu(z), \sigma^2 I)$$
3. **True Posterior (Intractable)**:
   $$p(z \mid x) = \frac{p_\theta(x \mid z) p(z)}{p_\theta(x)} = \frac{p_\theta(x \mid z) p(z)}{\int p_\theta(x \mid z) p(z) \, dz}$$
   The marginal likelihood $p_\theta(x)$ requires integrating over all possible latent states $z$, which is computationally intractable in high dimensions.
4. **Variational Inference (Encoder)**:
   We introduce an inference network $q_\phi(z \mid x)$ to approximate the true posterior $p(z \mid x)$:
   $$q_\phi(z \mid x) = \mathcal{N}\left(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x))\right)$$

---

## 3. Mathematical Derivation of the Evidence Lower Bound (ELBO)

We want to maximize the log marginal data likelihood $\log p_\theta(x)$.

Using Jensen's inequality or KL divergence identity:
$$D_{\text{KL}}\left(q_\phi(z \mid x) \,\|\, p(z \mid x)\right) = \int q_\phi(z \mid x) \log \frac{q_\phi(z \mid x)}{p(z \mid x)} \, dz \ge 0$$

Expanding the log fraction:
$$\begin{aligned}
D_{\text{KL}}\left(q_\phi(z \mid x) \,\|\, p(z \mid x)\right) &= \mathbb{E}_{q_\phi}\left[\log q_\phi(z \mid x) - \log p(x, z) + \log p(x)\right] \\
&= \mathbb{E}_{q_\phi}\left[\log q_\phi(z \mid x) - \log p_\theta(x \mid z) - \log p(z)\right] + \log p(x) \\
&= -\mathbb{E}_{q_\phi}\left[\log p_\theta(x \mid z)\right] + D_{\text{KL}}\left(q_\phi(z \mid x) \,\|\, p(z)\right) + \log p(x)
\end{aligned}$$

Rearranging for $\log p(x)$:
$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right]}_{\text{Reconstruction Quality}} - \underbrace{D_{\text{KL}}\left(q_\phi(z \mid x) \,\|\, p(z)\right)}_{\text{Regularizer to Standard Normal}} + \underbrace{D_{\text{KL}}\left(q_\phi(z \mid x) \,\|\, p(z \mid x)\right)}_{\ge 0}$$

Since $D_{\text{KL}} \ge 0$, we define the **Evidence Lower Bound (ELBO)** $\mathcal{L}_{\text{ELBO}}(\theta, \phi; x)$:
$$\log p(x) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] - D_{\text{KL}}\left(q_\phi(z \mid x) \,\|\, p(z)\right)$$

To minimize loss via gradient descent, we minimize the negative ELBO:
$$\mathcal{L}_{\text{VAE}}(\theta, \phi) = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$$

---

### 3.1 Closed-Form KL Divergence for Gaussian Distributions

When both $q_\phi(z \mid x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$ are multivariate Gaussians of dimension $d_z$, the KL divergence can be computed analytically in closed form without Monte-Carlo sampling:

$$D_{\text{KL}}\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I)\right) = -\frac{1}{2} \sum_{j=1}^{d_z} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

#### Proof:
$$\begin{aligned}
D_{\text{KL}} &= \int q(z) \log \frac{q(z)}{p(z)} \, dz = \mathbb{E}_{q}\left[\log q(z) - \log p(z)\right] \\
\log q(z) &= -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^d \log \sigma_j^2 - \frac{1}{2} \sum_{j=1}^d \frac{(z_j - \mu_j)^2}{\sigma_j^2} \\
\log p(z) &= -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^d z_j^2 \\
\mathbb{E}_q\left[\log q(z)\right] &= -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^d \log \sigma_j^2 - \frac{d}{2} \\
\mathbb{E}_q\left[\log p(z)\right] &= -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^d \mathbb{E}_q[z_j^2] = -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^d (\mu_j^2 + \sigma_j^2)
\end{aligned}$$

Subtracting yields the celebrated formula:
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d_z} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right) \quad \blacksquare$$

---

## 4. The Reparameterization Trick

```mermaid
flowchart LR
    subgraph Naive Sampling -- Gradient Blocked
        N_X[Input x] --> N_E[mu, sigma]
        N_E --> N_S[Sample z ~ N mu, sigma]
        N_S --> N_D[Decoder]
        N_S -.x Cannot Backprop .-> N_E
    end

    subgraph Reparameterization Trick -- Pathwise Gradient Flow
        R_X[Input x] --> R_E[mu, sigma]
        EPS[eps ~ N 0, I] --> R_Z[z = mu + sigma * eps]
        R_E --> R_Z
        R_Z --> R_D[Decoder]
        R_D ==>|dz| R_Z
        R_Z ==>|dz| R_E
    end
```

If we sample $z \sim q_\phi(z \mid x)$ directly, the sampling operation is a stochastic node with no continuous gradient with respect to encoder parameters $\phi$:
$$\nabla_\phi \mathbb{E}_{z \sim q_\phi}[f(z)] = \text{High-variance score function / REINFORCE}$$

The **Reparameterization Trick** isolates the randomness into an independent auxiliary standard normal noise variable $\epsilon \sim \mathcal{N}(0, I)$:
$$z = g(\phi, x, \epsilon) = \mu(x) + \sigma(x) \odot \epsilon$$

Now, $z$ is a deterministic, differentiable function of $\mu$ and $\sigma$:
$$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \log \sigma^2} = \frac{1}{2} \sigma \odot \epsilon = \frac{1}{2} (z - \mu)$$
Backpropagation proceeds through the computational graph with minimal variance!

---

## 5. Conditional VAE (C-VAE) for Controlled Predictive Trajectories

In industrial operations, we need to generate future trajectories *conditioned* on operating health states $y \in \{0, 1, 2\}$:

$$\begin{aligned}
\text{Encoder:} \quad & q_\phi(z \mid x, y) = \mathcal{N}\left(\mu_\phi([x \,\|\, y]), \sigma_\phi^2([x \,\|\, y])\right) \\
\text{Prior:} \quad & p(z) = \mathcal{N}(0, I) \\
\text{Decoder:} \quad & p_\theta(x \mid z, y) = g_\theta([z \,\|\, y])
\end{aligned}$$

This allows targeted predictive simulation:
- *"What will the temperature and vibration waveforms look like over the next 24 hours if the machine develops a Warning (State 1) versus a Catastrophic Fault (State 2)?"*

---

## 6. Generative Models Comparison

| Generative Paradigm | Sampling Latency | Training Stability | Likelihood Evaluation | Uncertainty Modeling |
|---|---|---|---|---|
| **Autoencoder (AE)** | Instant ($\mathcal{O}(1)$) | High | None (No density) | Deterministic only |
| **Variational Autoencoder (VAE)** | Instant ($\mathcal{O}(1)$) | Very High (ELBO objective) | Lower Bound (ELBO) | Closed-form Gaussian + MC Draws |
| **Generative Adversarial Net (GAN)** | Instant ($\mathcal{O}(1)$) | Low (Minimax instability) | None | Heuristic |
| **Diffusion (DDPM — Stage 8)** | Iterative ($\mathcal{O}(T)$ steps) | High (MSE noise matching) | Exact / Upper Bound | Rich Multi-modal Trajectories |
| **Flow Matching (Stage 10)** | Fast ODE integration | Very High | Exact via change-of-vars | Continuous probability paths |
