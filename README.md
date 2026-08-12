# 🎨 Day 71 — Conditional GAN: Class-Controlled Image Generation on CIFAR-10

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-CIFAR--10-EE4C2C?style=flat-square)
![GAN](https://img.shields.io/badge/CGAN-Conditional%20GAN-8B00FF?style=flat-square)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-20BEFF?style=flat-square)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2071-blueviolet?style=flat-square)

**Adding class labels doesn't automatically make a GAN generate high-quality images. CIFAR-10 is significantly more challenging than MNIST — diverse objects, backgrounds, and visual patterns make it a serious benchmark.**

</div>

---

## 📌 Overview

Day 64 built a vanilla GAN — it could generate Fashion MNIST images, but had no control over *which* class to generate. **Conditional GANs (CGANs)** solve this by feeding class labels into both the Generator and Discriminator, giving explicit control over what the network produces.

Ask the Generator for a cat → it generates a cat. Ask for a ship → it generates a ship. The same architecture, with one critical addition: **conditioning information flows through both networks**.

> **Hard truth learned today:** Adding class labels doesn't automatically make a GAN generate high-quality images. CIFAR-10 is significantly more challenging than simple datasets like MNIST because the images are more complex and contain diverse objects, backgrounds, and visual patterns.

---

## 🧠 Vanilla GAN vs Conditional GAN

```
VANILLA GAN (Day 64):
  z ~ N(0,1) ─────────────► Generator ──► Fake Image
                                               │
  Real Images ──────────────────────────► Discriminator ──► Real / Fake
                                               │
              No control over what gets generated ❌

CONDITIONAL GAN (Day 71):
  z ~ N(0,1) ──┐
               ├──► Generator ──► Fake Image (class-specific)
  Class Label ─┘                      │
                                      │
  Real Images ────────┐               │
                      ├──► Discriminator ──► Real / Fake
  Class Label ────────┘
                              │
       Control: "Generate a Cat" → Generator produces cat images ✅
```

**The conditional information answers two questions:**
- **Generator:** "What class of image should I create?"
- **Discriminator:** "Does this image actually look like the class it claims to be?"

---

## 📊 Dataset — CIFAR-10

| Property | Detail |
|---|---|
| Source | `torchvision.datasets.CIFAR10` |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32×32 RGB (3 channels) |
| Classes | 10 (balanced, 6000/class) |
| Complexity | Significantly harder than MNIST — real-world objects with varied backgrounds |

### 10 Classes

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | ✈️ Airplane | 5 | 🐶 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚚 Truck |

---

## 🏗️ CGAN Architecture

### How Class Labels Are Injected

```
CLASS LABEL EMBEDDING:
  Label (integer 0–9)
       │
  nn.Embedding(num_classes=10, embed_dim=50)
       │
  50-dimensional embedding vector
       │
  Concatenated with noise z (Generator)
  OR flattened image (Discriminator)
```

### Generator

```
Inputs:
  z     ~ N(0,1)  shape: (batch, 100)   ← random noise
  label            shape: (batch,)       ← class index (e.g. 3 = Cat)
         │
  Embedding(label) → (batch, 50)
         │
  Concatenate [z, label_embed] → (batch, 150)
         │
  Linear(150 → 256)  + LeakyReLU + BatchNorm
         │
  Linear(256 → 512)  + LeakyReLU + BatchNorm
         │
  Linear(512 → 1024) + LeakyReLU + BatchNorm
         │
  Linear(1024 → 3×32×32)  + Tanh
         │
  Reshape → (batch, 3, 32, 32)
         │
  Generated CIFAR-10 image conditioned on class label
```

### Discriminator

```
Inputs:
  image   shape: (batch, 3, 32, 32) → flatten → (batch, 3072)
  label   shape: (batch,)
         │
  Embedding(label) → (batch, 50)
         │
  Concatenate [image_flat, label_embed] → (batch, 3122)
         │
  Linear(3122 → 1024) + LeakyReLU + Dropout(0.3)
         │
  Linear(1024 → 512)  + LeakyReLU + Dropout(0.3)
         │
  Linear(512  → 256)  + LeakyReLU + Dropout(0.3)
         │
  Linear(256  → 1)    + Sigmoid
         │
  P(image is real AND matches the label) ∈ (0, 1)
```

---

## 🔬 What I Implemented

### 1. Data Loading

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

# ─── Transforms ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),    # normalize each RGB channel
        std =(0.5, 0.5, 0.5)     # → pixel values ∈ [-1, 1]
    )
])

dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
loader  = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

print(f"Dataset size : {len(dataset)}")
print(f"Image shape  : {dataset[0][0].shape}")   # (3, 32, 32)
```

### 2. Generator

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10,
                 embed_dim=50, img_channels=3, img_size=32):
        super(ConditionalGenerator, self).__init__()

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        input_dim        = noise_dim + embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()    # output ∈ [-1, 1] matches normalization
        )

        self.img_shape = (img_channels, img_size, img_size)

    def forward(self, z, labels):
        label_embedded = self.label_embed(labels)           # (batch, 50)
        x              = torch.cat([z, label_embedded], dim=1)  # (batch, 150)
        img_flat       = self.net(x)
        return img_flat.view(img_flat.size(0), *self.img_shape)  # (batch, 3, 32, 32)
```

### 3. Discriminator

```python
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=50,
                 img_channels=3, img_size=32):
        super(ConditionalDiscriminator, self).__init__()

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        input_dim        = img_channels * img_size * img_size + embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat       = img.view(img.size(0), -1)              # (batch, 3072)
        label_embedded = self.label_embed(labels)               # (batch, 50)
        x              = torch.cat([img_flat, label_embedded], dim=1)  # (batch, 3122)
        return self.net(x)
```

### 4. Conditional Training Loop

```python
# ─── Setup ───────────────────────────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISE_DIM = 100
N_CLASSES = 10

G = ConditionalGenerator(noise_dim=NOISE_DIM).to(device)
D = ConditionalDiscriminator().to(device)

criterion = nn.BCELoss()
opt_G     = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D     = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Fixed noise + one sample per class for consistent visualization
fixed_noise  = torch.randn(N_CLASSES, NOISE_DIM, device=device)
fixed_labels = torch.arange(N_CLASSES, device=device)   # [0,1,2,...,9]

EPOCHS   = 100
g_losses = []
d_losses = []

for epoch in range(EPOCHS):
    g_epoch, d_epoch = 0.0, 0.0

    for real_imgs, labels in loader:
        batch_size  = real_imgs.size(0)
        real_imgs   = real_imgs.to(device)
        labels      = labels.to(device)

        real_targets = torch.ones(batch_size, 1, device=device)
        fake_targets = torch.zeros(batch_size, 1, device=device)

        # ── Train Discriminator ────────────────────────────────────────────────
        opt_D.zero_grad()

        # Real images with correct labels → D should output 1
        d_real   = D(real_imgs, labels)
        loss_real = criterion(d_real, real_targets)

        # Fake images with same labels → D should output 0
        z         = torch.randn(batch_size, NOISE_DIM, device=device)
        fake_imgs = G(z, labels).detach()
        d_fake    = D(fake_imgs, labels)
        loss_fake = criterion(d_fake, fake_targets)

        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        opt_D.step()

        # ── Train Generator ────────────────────────────────────────────────────
        opt_G.zero_grad()

        z         = torch.randn(batch_size, NOISE_DIM, device=device)

        # Use RANDOM labels — G must learn to generate ANY class
        rand_labels = torch.randint(0, N_CLASSES, (batch_size,), device=device)
        fake_imgs   = G(z, rand_labels)

        # G wants D to think fakes are real (given the label)
        d_output = D(fake_imgs, rand_labels)
        g_loss   = criterion(d_output, real_targets)

        g_loss.backward()
        opt_G.step()

        g_epoch += g_loss.item()
        d_epoch += d_loss.item()

    g_losses.append(g_epoch / len(loader))
    d_losses.append(d_epoch / len(loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | G: {g_losses[-1]:.4f} | D: {d_losses[-1]:.4f}")
        visualize_conditional(G, fixed_noise, fixed_labels, epoch + 1)
```

### 5. Class-Conditional Visualization

```python
def visualize_conditional(generator, noise, labels, epoch):
    """Generate one image per CIFAR-10 class and display side by side."""
    generator.eval()
    with torch.no_grad():
        imgs = generator(noise, labels).cpu()
        imgs = (imgs + 1) / 2    # denormalize [-1,1] → [0,1]
        imgs = imgs.clamp(0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(imgs, axes)):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(CIFAR_CLASSES[i], fontsize=10)
        ax.axis('off')

    plt.suptitle(f'CGAN Generated CIFAR-10 — Epoch {epoch}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'outputs/epoch_{epoch:03d}_classes.png', dpi=100)
    plt.close()
    generator.train()
```

---

## 🆚 Vanilla GAN vs CGAN — Side by Side

| Aspect | Vanilla GAN (Day 64) | Conditional GAN (Day 71) |
|---|---|---|
| Generator input | Noise z only | Noise z + class label |
| Discriminator input | Image only | Image + class label |
| Output control | ❌ Random class | ✅ Specific class on demand |
| Architecture change | — | Add `nn.Embedding` to both G and D |
| Label injection | — | Concatenate embedding to input |
| Training | Single objective | Same — just conditioned |
| Use case | Unconditional generation | Class-conditional, targeted generation |

---

## 🔄 CGAN Loss Functions

**Discriminator** must now judge: is this image **both real AND correctly labeled?**

$$\mathcal{L}_D = -\mathbb{E}[\log D(x, y)] - \mathbb{E}[\log(1 - D(G(z, y), y))]$$

**Generator** must fool D into accepting its output for a specific class:

$$\mathcal{L}_G = -\mathbb{E}[\log D(G(z, y), y)]$$

Where $y$ = class label (conditioned information)

---

## 📈 Why CIFAR-10 is Harder Than MNIST/Fashion-MNIST

| Factor | MNIST | Fashion-MNIST | CIFAR-10 |
|---|---|---|---|
| Image size | 28×28 | 28×28 | 32×32 |
| Channels | 1 (grayscale) | 1 (grayscale) | 3 (RGB) |
| Pixels per image | 784 | 784 | 3,072 |
| Background variation | ❌ Uniform | ❌ Uniform | ✅ Diverse |
| Object pose variation | Low | Low | High |
| Intra-class variation | Low | Medium | High |
| Classes confused | Rarely | Sometimes | Frequently |
| Epochs to reasonable output | ~20–30 | ~30–50 | ~100–200+ |

---

## ⚠️ CGAN Training Challenges

| Challenge | Detail | Mitigation |
|---|---|---|
| **Mode collapse per class** | G generates same image for all inputs of a class | Minibatch discrimination, spectral norm |
| **Label-image mismatch** | G ignores label and generates whatever fools D | Ensure both G and D receive same label |
| **CIFAR complexity** | 3-channel 32×32 images much harder than grayscale | Use DCGAN (Conv layers) instead of FC |
| **Training instability** | D dominates early, G can't learn | Match learning rates, use label smoothing |
| **Slow convergence** | CIFAR needs many more epochs | Use progressive growing, residual connections |

---

## 🔁 GAN Evolution — Where CGAN Fits

```
Vanilla GAN (Day 64)
  │  ← Unconditional, FC layers, unstable on complex images
  ▼
Conditional GAN — CGAN (Day 71)
  │  ← Class labels control generation, same FC architecture
  ▼
DCGAN (Deep Convolutional GAN)
  │  ← Conv/ConvTranspose layers, much more stable on CIFAR
  ▼
Pix2Pix / CycleGAN
  │  ← Image-to-image translation (sketch → photo, etc.)
  ▼
BigGAN
  │  ← Large-scale conditional generation, class conditioning at scale
  ▼
StyleGAN / StyleGAN2
       ← State of the art, photorealistic, fine-grained style control
```

---

## 💡 Key Learnings

- **CGANs extend GANs with one addition** — embedding the class label and concatenating it to both G and D inputs; the rest of the adversarial training loop is identical
- **Both G and D must receive the same label** — if only G sees the label, D can't penalize label-image mismatches; conditioning breaks down
- **CIFAR-10 is a real benchmark** — 3-channel RGB images with object + background variation make generation fundamentally harder than grayscale digit datasets
- **`nn.Embedding` is the standard conditioning mechanism** — maps integer class indices to dense vectors that G and D can learn from
- **Fully connected CGAN on CIFAR is a known limitation** — convolutional architectures (DCGAN-style) produce significantly better results on complex RGB datasets

---

## 🗂️ Project Structure

```
day-71-cgan-cifar10/
├── train.py                     # Full CGAN training loop
├── generator.py                 # ConditionalGenerator class
├── discriminator.py             # ConditionalDiscriminator class
├── visualize.py                 # Per-class image grid visualization
│
├── data/                        # Auto-downloaded CIFAR-10
│
├── outputs/
│   ├── epoch_010_classes.png    # 10-class grid at epoch 10
│   ├── epoch_050_classes.png    # 10-class grid at epoch 50
│   ├── epoch_100_classes.png    # 10-class grid at epoch 100
│   └── loss_curves.png
│
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-71-cgan-cifar10
cd day-71-cgan-cifar10
pip install -r requirements.txt
python train.py
```

**Requirements:**
```
torch
torchvision
numpy
matplotlib
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 71 of 100 — Conditional GAN: Class-Controlled Image Generation on CIFAR-10

| ← Previous | Current | Next → |
|---|---|---|
| [Day 70 — KNN Multiclass](#) | **Day 71 — CGAN CIFAR-10** | [Day 72](#) |



---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #CGAN #ConditionalGAN #CIFAR10 #PyTorch #GenerativeAI #DeepLearning #ComputerVision</sub>
</div>
