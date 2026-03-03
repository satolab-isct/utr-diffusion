# UTR-Diffusion

Official implementation of the paper:

> UTR-Diffusion: Conditional Diffusion Modeling for Multi-objective and Constrained UTR Design  
> ECCB 2026 Proceedings

UTR-Diffusion is a diffusion-based generative framework for 5′ UTR sequence design that enables:

- Continuous control of translation-related indicators:
  - Mean Ribosome Load (MRL)
  - Minimum Free Energy (MFE)
- Explicit codon- or amino-acid constraints at user-specified positions
- Multi-objective conditional generation

---

## 🔬 Overview

UTR-Diffusion supports:

- Unconditional generation
- Continuous conditional generation (MRL, MFE)
- Constrained generation (codon / amino acid clamping)
- Multi-label generation (MRL + MFE)
- Evaluation pipeline (MRL/MFE prediction)

---

## 📦 Installation

```bash
git clone https://github.com/satolab-isct/utr-diffusion
cd utr-diffusion
pip install -r requirements.txt
