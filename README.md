# Skin Disease Classification

Hey! This is a deep learning project that identifies skin diseases from photos. Upload an image of a skin lesion, and it'll tell you what it might be.

**Current accuracy: 81.6%** on 7 different skin conditions.

## What It Does

You give it a photo of a skin lesion, it tells you which of these conditions it most likely is:

- Basal cell carcinoma
- Benign keratosis  
- Dermatofibroma
- Healthy skin
- Measles
- Melanocytic nevi (moles)
- Melanoma

The model uses EfficientNetV2S (a modern image classification network) trained in two phases — first learning general features, then fine-tuning specifically for skin lesions.

## Quick Start

### 1. Set Up (WSL2 recommended on Windows)

```bash
# In WSL2 Ubuntu terminal
cd /mnt/d/Code/Skin-disease-prediction
python3 -m venv venv
source venv/bin/activate
pip install "tensorflow[and-cuda]"
pip install -r requirements.txt
```

### 2. Get the Dataset

```bash
python download_dataset.py
python simple_rebalance.py
```

### 3. Train the Model

```bash
python train.py
```

Takes about 2-3 hours on an RTX 3060. The best model saves automatically.

### 4. Test It

```bash
python test.py
```

### 5. Run the Web App

```bash
python app.py
```

Then open http://localhost:5000 in your browser. Drag and drop a skin image to get predictions.

## Project Files

| File | What it does |
|------|--------------|
| `train.py` | Trains the model (two-phase transfer learning) |
| `test.py` | Evaluates model accuracy, saves confusion matrix |
| `app.py` | Flask web app for predictions |
| `download_dataset.py` | Downloads skin lesion images from Hugging Face |
| `simple_rebalance.py` | Balances the dataset (some classes have way more images) |

See [structure.md](structure.md) for the full file layout.

## Results

| Metric | Score |
|--------|-------|
| Overall accuracy | 81.6% |
| Macro F1 | 0.827 |
| Weighted F1 | 0.819 |

The model is especially good at detecting Melanoma (99% F1) and Healthy skin (99% F1). It struggles a bit more with Benign keratosis (57% F1) which often looks similar to other conditions.

## Requirements

- **GPU**: NVIDIA with 4GB+ VRAM (trained on RTX 3060 Laptop)
- **RAM**: 8GB minimum
- **Python**: 3.10
- **OS**: Ubuntu 22.04 (WSL2 works great on Windows)

TensorFlow 2.20+ only supports GPU on Linux/WSL2. If you're on native Windows, you'll need to use CPU or an older TensorFlow version.

## Common Issues

**"GPU not found"**
- Make sure you're in WSL2, not PowerShell
- Check `nvidia-smi` works in WSL
- Reinstall TensorFlow: `pip install "tensorflow[and-cuda]"`

**"Out of memory"**  
- Edit `train.py` and set `self.batch_size = 16`

**"balanced_dataset not found"**
- Run `python simple_rebalance.py` first

**Slow training**
- Move the project to WSL's native filesystem for 3-5x faster I/O:
  ```bash
  cp -r /mnt/d/Code/Skin-disease-prediction ~/skin-project
  cd ~/skin-project
  ```

## How It Works

1. **EfficientNetV2S backbone** — pretrained on ImageNet, already knows how to see textures, edges, and patterns
2. **Phase 1** — freeze the backbone, train only the classification head (40 epochs)
3. **Phase 2** — unfreeze top 100 layers, fine-tune everything together (30 epochs)
4. **Learning rate** — starts low, warms up, then slowly decays (cosine schedule)
5. **Class balancing** — weights rare classes higher so the model doesn't just predict the common ones

### Supported Classes (7-Class Model)
This model was trained on a balanced subset of 7 specific skin conditions to maximize accuracy and reduce class imbalance. 

The network is capable of classifying the following conditions:
1. **Basal cell carcinoma**
2. **Benign keratosis** 
3. **Healthy** (No visible disease)
4. **HFMD** (Hand, Foot, and Mouth Disease)
5. **Melanocytic nevi**
6. **Melanoma** 
7. **Monkeypox** 

The model expects raw images (0-255 pixel values) and handles normalization internally.

---

