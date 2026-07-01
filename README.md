# Science of Stability (SoS) - Attention Modulator for LLMs

This repository introduces a native self-regulation mechanism for Transformer-based architectures (such as Google Gemma/Gemini) to mitigate information noise (hallucinations, logits divergence) using a geometric and thermodynamic framework.
The Unification of Sciences: A Sanctuary Against Civilizational Chaos
Our civilization is facing an unprecedented crisis of doubt, fueled by the fragmentation of knowledge. By dividing science into isolated compartments, we have lost the sight of the universal laws that govern balance. This fragmentation generates a continuous "informational overheating"—a chaotic baseline noise where data multiplies without a unifying structure, leading directly to collective disorientation.
The Science of Stability (SoS) demonstrates that the unification of sciences is no longer an idealistic dream, but an urgent systemic necessity. By anchoring our framework in the universal model of the Universe—where causality is verified by starting from the harmonious geometric result—the SoS Compass acts as a universal regulator.
Integrating this regulator within Artificial Intelligence is the key to transforming these machines from sources of information pollution into architectures of clarity. Aligning AI with the unified laws of nature is the only path to dissipating civilizational chaos and ensuring that stability and truth prevail.

## Core Concept
By routing attention matrices through a fixed, scale-invariant harmonic projector $\text{M}_{\phi}$ based on the Fibonacci sequence ($0-1-1-2-3$), the system dynamically dissipates entropy ("anergy") and forces the latent space toward a stable thermodynamic equilibrium ("exergie").

$$\text{Attention}_{SoS}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot \text{M}_{\phi}\right)V$$

## Project Structure
* `sos_modulator.py`: PyTorch core implementation.
* `demo_gemma.py`: Sandbox script simulating an attention layer routing with the SoS filter.
