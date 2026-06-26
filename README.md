# Science of Stability (SoS) - Attention Modulator for LLMs

This repository introduces a native self-regulation mechanism for Transformer-based architectures (such as Google Gemma/Gemini) to mitigate information noise (hallucinations, logits divergence) using a geometric and thermodynamic framework.

## Core Concept
By routing attention matrices through a fixed, scale-invariant harmonic projector $\text{M}_{\phi}$ based on the Fibonacci sequence ($0-1-1-2-3$), the system dynamically dissipates entropy ("anergy") and forces the latent space toward a stable thermodynamic equilibrium ("exergie").

$$\text{Attention}_{SoS}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot \text{M}_{\phi}\right)V$$

## Project Structure
* `sos_modulator.py`: PyTorch core implementation.
* `demo_gemma.py`: Sandbox script simulating an attention layer routing with the SoS filter.