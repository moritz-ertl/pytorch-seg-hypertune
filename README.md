# Hypertune Segmentation Networks with PyTorch

This repository provides a flexible framework for **hyperparameter tuning of semantic segmentation networks**. It combines segmentation models, adaptive augmentation, and hyperparameter search in one streamlined pipeline.

---

## 🚀 Features
- **Segmentation Networks**: Based on [segmentation_models.pytorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch)  
- **Hyperparameter Optimization**: Powered by [Optuna](https://optuna.org/) for efficient search strategies (TPE, grid, random search, pruning)  
- **Tunable Adaptive Augmentation**: Inspired by *Hou et al.* ([Monotonic Curriculum, 2023](https://arxiv.org/abs/2309.04747)) – progressively introduces stronger augmentations as training progresses  
- **PyTorch Lightning** integration for clean training loops and logging  
- **Config-driven design**: Define search spaces, training params, and augmentations via YAML configs  
- **Debug-friendly dataset class** with adaptive augmentation hooks  

---

## 📂 Repository Structure
```
├── configs/
│   └── default.yaml
├── src/
│   ├── aug/
│   │   └── adaptive.py
│   ├── data/
│   │   ├── datamodule.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── factory.py
│   │   └── lightning_module.py
│   ├── utils/
│   │   └── search_space.py
│   ├── train.py
│   └── tune.py
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation
```bash
# Clone repo
git clone https://github.com/ertlmoritz/pytorch-seg-hypertune.git
cd pytorch-seg-hypertune

# Option A: Create and activate virtual environment using venv
python -m venv .venv

# Activate
source .venv/bin/activate        # On Linux/Mac
.venv\Scripts\activate           # On Windows

# Option B: Create and activate virtual environment using conda
conda create -n segm python=3.10
conda activate segm

# Install dependencies (works for both venv or conda)
pip install -r requirements.txt
```

**Key Dependencies**:
- [PyTorch](https://pytorch.org/)  
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)  
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)  
- [Optuna](https://optuna.org/)  
- [Albumentations](https://albumentations.ai/)  

---

## 🧑‍💻 Usage
### Train a model
```bash
python src/train.py --config configs/default.yaml
```

### Run hyperparameter search
```bash
python src/tune.py --config configs/default.yaml --trials 50
```

**Configurable parameters include**:
- Encoder architecture (`resnet34`, `efficientnet-b0`, …)  
- Decoder type (`Unet`, `FPN`, `DeepLabV3+`)  
- Loss functions (`DiceLoss`, `CrossEntropy`, `FocalLoss`)  
- Learning rate, optimizer, scheduler  
- Augmentation strategy (static vs. adaptive curriculum)  

---

## 📖 Adaptive Augmentation (Hou et al., 2023)
This repo implements an **epoch-dependent curriculum augmentation**, following:

> *Monotonic curriculum which progressively introduces more augmented samples as the training epoch increases.*  
> — Hou et al., [Paper](https://arxiv.org/abs/2309.04747)

- Early epochs → lighter augmentations  
- Later epochs → stronger augmentations  
- Controlled by `tau` parameter in the config  

---

## 📊 Example Workflow
1. Define dataset paths and hyperparameter search space in `configs/default.yaml`   
2. Run `tune.py` to explore hyperparameters  
3. Monitor results via Optuna dashboard:
   ```bash
   optuna-dashboard sqlite:///optuna.db
   ```
4. Run `train.py` with optimized hyperparameters 

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) by Pavel Yakubovskiy  
- [Optuna](https://optuna.org/) for hyperparameter optimization  
- [Albumentations](https://albumentations.ai/) for data augmentation  
- [Hou et al., 2023](https://arxiv.org/abs/2309.04747): Inspiration for adaptive augmentation curriculum
