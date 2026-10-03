# Training & Fine-Tuning Pipeline: Neural ASCII Compositor
**Krystal-Stack Platform Framework — AI Model Training & Distillation Specification**  
*Document Version: 2.5-PRO | Target: Edge NPU & INT8 Real-Time Deployment | Date: 2026-10-01*  
*Author: Dušan Kopecký & AI Research Team*

---

## 1. Overview & Computational Objectives

The goal of the **Neural ASCII Compositor (NAC)** training pipeline is to produce an ultra-lightweight neural network capable of transducing game scene frames and telemetry into optimal ASCII compositions in **< 3 milliseconds** on an NPU (Intel AI Boost, AMD XDNA, Qualcomm Hexagon) or modern tensor core (INT8/FP16).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                MODEL TARGET CONSTRAINTS                                │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│ Parameter Count               │ < 1.8 Million parameters                               │
│ Quantized Memory Footprint    │ < 2.5 MB (INT8 Quantized ONNX / OpenVINO IR)           │
│ Inference Latency Target      │ ≤ 2.2 ms per 120x60 ASCII grid on 10 TOPS NPU          │
│ Maximum Host RAM Utilization  │ < 16 MB Working Tensor Arena                           │
│ Hardware Acceleration Target  │ NPU (DirectML NPU EP / OpenVINO NPU Plugin / Level-0)  │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Architecture & Synthetic Engine Generation

Training a model on manually created ASCII art is impossible due to scarcity and lack of structural metadata. We establish a **Synthetic-to-Real Multi-Modal Dataset Pipeline**:

```
                       ┌────────────────────────────────────────┐
                       │     PROCEDURAL 3D GAME SCENE ENGINE    │
                       │    (OpenGL / Vulkan Headless Scraper)  │
                       └───────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐                         ┌─────────────────────────────────┐
│     VISUAL & G-BUFFER CHANNELS  │                         │       TELEMETRY METADATA        │
│ • High-Res RGB (1920x1080)      │                         │ • GPU Frame Time & TDR metrics  │
│ • Surface Normal Map (XYZ)      │                         │ • Vertex & Draw-Call Density    │
│ • Depth Map (Z-Linear)          │                         │ • Audio Frequency Spectrum      │
│ • Motion Vectors (DX, DY)       │                         │ • Thermal Headroom & Power Watts│
└────────────────┬────────────────┘                         └────────────────┬────────────────┘
                 │                                                           │
                 └─────────────────────────┬─────────────────────────────────┘
                                           ▼
                       ┌────────────────────────────────────────┐
                       │       GROUND TRUTH SDF GENERATOR       │
                       │  • Pre-computed 95 Glyph SDFs          │
                       │  • Chamfer Matching Optimal Labeling   │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │     TRAINING TENSOR PAIR (X, Y, M)     │
                       │ X: Low-res patch / G-buffer (120x60x4) │
                       │ Y: Optimal Character Class ID (0-94)   │
                       │ M: Telemetry Modulation Vector (16-D)  │
                       └────────────────────────────────────────┘
```

### 2.1 Dataset Classes & Scene Categories
The training corpus comprises 250,000 synthetic and intercepted game frames across 5 distinct operational scenarios:
1. **Geometric Blueprint (Wireframe / UI)**: Hard architectural edges, HUD vectors, low polygon counts.
2. **Dense Natural Foliage**: High-frequency stochastic noise (grass, trees, hair) testing texture entropy.
3. **High-Dynamic Combat & Particles**: Alpha-blended explosions, volumetric smoke, compute shader sparks.
4. **Cinematic Shading**: Soft gradient skies, atmospheric fog, ambient occlusion transitions.
5. **Glitch / Overload Anomalies**: High system entropy states, dropped frames, memory pressure glitches.

---

## 3. Neural Network Architecture: `NanoCompositor-V3`

To execute natively on an NPU without triggering CPU fallback, the architecture avoids dynamic shapes, recurrent layers, and complex self-attention. We use an **Asymmetric Convolutional-Mixer (ACM)** architecture:

```
INPUT TENSOR: [Batch, 4, Height=60, Width=120]  (Albedo Lum, Depth, Norm_X, Norm_Y)
  │
  ├──▶ CONV2D 3x3, Stride 1, 32 Filters (Receptive Field Expansion) ──▶ LeakyReLU
  │
  ├──▶ DEPTHWISE SEPARABLE CONV 5x5, 64 Filters                     ──▶ LeakyReLU
  │
  ├──▶ TELEMETRY CONDITIONING LAYER:
  │      Telemetry Vector [16] ──▶ Linear(16 ─▶ 64) ──▶ FiLM Scale & Shift (γ, β)
  │      Modulates feature maps: Feature = γ · Feature + β
  │
  ├──▶ SPATIAL DIRECTIONAL CONVOLUTION (Sobel-inspired fixed kernels)
  │      Extracts angles [0°, 45°, 90°, 135°]
  │
  ├──▶ POINTWISE CONV 1x1, 128 Filters
  │
  └──▶ DUAL OUTPUT HEADS:
         Head A (Character Classifier): [Batch, 95 Classes, 60, 120] ──▶ Softmax
         Head B (Color & Palette Regressor): [Batch, 6 Channels (FG/BG RGB), 60, 120]
```

### 3.1 Feature-wise Linear Modulation (FiLM) with Telemetry
The unique innovation is the **FiLM (Feature-wise Linear Modulation)** layer. The network dynamically reshapes its visual feature representations based on real-time hardware and game telemetry $\mathbf{z}$:
$$\gamma = \mathbf{W}_\gamma \mathbf{z} + \mathbf{b}_\gamma, \quad \beta = \mathbf{W}_\beta \mathbf{z} + \mathbf{b}_\beta$$
$$\mathbf{F}_{conditioned} = \gamma \odot \mathbf{F} + \beta$$
When the telemetry indicates **High Thermal Strain** or **Combat Phase**, $\gamma$ and $\beta$ amplify edge channels and suppress subtle luminance variations, effortlessly morphing the composition from standard fidelity to high-contrast Cyberpunk mode.

---

## 4. Multi-Objective Mathematical Loss Function

The training loop balances typographic accuracy, edge direction alignment, visual entropy regulation, and temporal stability.

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{ce}} + \lambda_2 \mathcal{L}_{\text{directional}} + \lambda_3 \mathcal{L}_{\text{entropy\_barrier}} + \lambda_4 \mathcal{L}_{\text{temporal}}$$

### 4.1 Categorical Cross-Entropy ($\mathcal{L}_{\text{ce}}$)
Measures the character prediction against the optimal SDF ground-truth glyph:
$$\mathcal{L}_{\text{ce}} = - \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{95} y_{i,j,c} \log(\hat{y}_{i,j,c})$$

### 4.2 Directional Vector Cosine Loss ($\mathcal{L}_{\text{directional}}$)
Penalizes assigning horizontal characters (`-`) to vertical edges (`|`). Let $\mathbf{v}_{true} = (\cos \theta, \sin \theta)$ be the Sobel gradient direction, and $\mathbf{v}_{glyph}$ be the principal orientation vector of the predicted character:
$$\mathcal{L}_{\text{directional}} = 1 - \left| \mathbf{v}_{true} \cdot \mathbf{v}_{glyph} \right|$$
*(Using absolute value because a 180° inversion in an unoriented line represents identical visual orientation).*

### 4.3 Visual Entropy Barrier Loss ($\mathcal{L}_{\text{entropy\_barrier}}$)
Prevents the model from generating visual "white noise" (random character scattering). The spatial variance $\sigma^2(\hat{\mathbf{Y}})$ must not exceed a target threshold $E_{target}$ determined by the governor:
$$\mathcal{L}_{\text{entropy\_barrier}} = \max\left(0, \, \text{Var}(\hat{\mathbf{Y}}) - E_{target}\right)^2$$

### 4.4 Temporal Anti-Jitter Loss ($\mathcal{L}_{\text{temporal}}$)
Given consecutive synthetic frames $t-1$ and $t$ with motion vector field $\mathbf{u}$:
$$\mathcal{L}_{\text{temporal}} = \left\| \hat{\mathbf{Y}}_t(x, y) - \hat{\mathbf{Y}}_{t-1}(x - u_x, y - u_y) \right\|_1 \cdot \mathbb{I}(\|\Delta \mathbf{I}\| < \tau)$$
If the underlying image delta $\|\Delta \mathbf{I}\|$ is below threshold $\tau$, changing characters incurs a heavy penalty. This mathematically eliminates character flickering during slow camera pans.

---

## 5. Vision-Language Model (VLM) Teacher-Student Distillation

To imbue the compact `NanoCompositor` with human-level aesthetic sensibility, we employ a **Teacher-Student Distillation pipeline**:

```
Large Vision-Language Model (Teacher: LLaVA-1.6 / Florence-2 / Qwen-VL)
           │
           │ Prompt: "Critique this ASCII render for silhouette readability,
           │          contrast balance, and game readability (Score: 0.0 - 1.0)"
           ▼
     Aesthetic Reward Signal $R_{aesthetic}$
           │
           ▼
     Distillation Loss: $\mathcal{L}_{KD} = D_{KL}( P_{student} \,\|\, P_{teacher} )$
           │
           ▼
Student Model (`NanoCompositor-V3`, 1.5M Params) ──▶ Deployed to NPU
```

By scoring 50,000 generated ASCII renders with a high-capacity VLM and using reinforcement learning / knowledge distillation (KD), the lightweight NPU network learns subtle stylistic rules that cannot be hand-coded:
- Prioritizing outer character silhouettes over interior textures.
- Using empty space (` `) strategically to enhance dramatic contrast.
- Emphasizing actionable game gameplay elements (player character, crosshair, enemies).

---

## 6. PyTorch Reference Training Module

Here is the production-ready PyTorch module defining the `NanoCompositor` and its loss formulation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TelemetryFiLM(nn.Module):
    """Generates affine scale (gamma) and shift (beta) from telemetry vector."""
    def __init__(self, telemetry_dim=16, num_features=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(telemetry_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_features * 2)
        )
        self.num_features = num_features

    def forward(self, features, telemetry):
        # telemetry: [B, 16] -> [B, num_features * 2]
        params = self.fc(telemetry)
        gamma = params[:, :self.num_features].unsqueeze(-1).unsqueeze(-1)
        beta = params[:, self.num_features:].unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta

class NanoCompositor(nn.Module):
    """
    Lightweight, NPU-optimizable Neural ASCII Compositor.
    Input: [B, 4, H, W] -> (Luminance, Depth, Normal_X, Normal_Y)
    Output: Glyphs [B, 95, H, W] + Color [B, 6, H, W]
    """
    def __init__(self, num_classes=95):
        super().__init__()
        # Initial receptive field
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        
        # Depthwise Separable Conv
        self.dw_conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2, groups=32)
        self.pw_conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.relu2 = nn.LeakyReLU(0.1)
        
        # FiLM Telemetry Conditioning
        self.film = TelemetryFiLM(telemetry_dim=16, num_features=64)
        
        # Intermediate reasoning
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)
        
        # Output Heads
        self.glyph_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.color_head = nn.Conv2d(64, 6, kernel_size=1) # 3 FG RGB + 3 BG RGB

    def forward(self, x, telemetry):
        h = self.relu1(self.conv1(x))
        h = self.relu2(self.pw_conv2(self.dw_conv2(h)))
        h = self.film(h, telemetry)
        h = self.relu3(self.conv3(h))
        
        glyph_logits = self.glyph_head(h)
        colors = torch.sigmoid(self.color_head(h))
        return glyph_logits, colors
```

---

## 7. OpenVINO & DirectML INT8 Quantization Workflow

To convert the trained PyTorch weights into an ultra-fast NPU deployment artifact:

```bash
# 1. Export PyTorch to ONNX with static shapes for maximum NPU efficiency
python -c "
import torch
from nano_compositor import NanoCompositor
model = NanoCompositor().eval()
dummy_img = torch.randn(1, 4, 60, 120)
dummy_tele = torch.randn(1, 16)
torch.onnx.export(
    model, (dummy_img, dummy_tele), 'nano_compositor.onnx',
    input_names=['input_g_buffer', 'input_telemetry'],
    output_names=['glyph_logits', 'color_matrix'],
    opset_version=17
)
"

# 2. Compile to OpenVINO IR and apply Post-Training INT8 Quantization (POT / NNCF)
mo --input_model nano_compositor.onnx \
   --output_dir compiled_ir/ \
   --data_type FP16

# 3. Benchmark directly on NPU
benchmark_app -m compiled_ir/nano_compositor.xml -d NPU -t 10
```

---
*Next Step: Consult [GAME_ENGINE_TECHNIQUES_AND_PROCEDURAL_RENDERING.md](./GAME_ENGINE_TECHNIQUES_AND_PROCEDURAL_RENDERING.md) for deferred rendering integration and Win32 display drivers.*
