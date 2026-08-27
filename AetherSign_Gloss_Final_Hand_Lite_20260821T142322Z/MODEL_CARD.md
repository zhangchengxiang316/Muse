# AetherSign Muse / Gloss Translator Final Submission

## 1. Task

The submitted Muse models perform six-class isolated sign-gloss classification.

- Logical input: `[1, 4, 54, 64]`, Float32
- Channels: normalized x, normalized y, validity, hand-slot identity
- Joints: 2 hands x 27 points = 54
- Temporal window: 64 frames
- Output: `[1, 6, 1, 1]` logits
- Class order: rain, long, short, go, thick, no_gesture

## 2. Submitted final variants

### Full Hand / Exp-A

- Upstream landmark frontend: `hand_final`
- Muse training: Multi-Stride
- Stride probabilities: s3=0.15, s4=0.40, s5=0.35, s6=0.10
- Explicit speed augmentation: disabled
- Start jitter: disabled
- Validation accuracy / Macro-F1: 99.074% / 99.073%
- Robustness-grid mean accuracy / Macro-F1: 94.306% / 94.168%
- Worst-cell accuracy / Macro-F1: 91.667% / 91.567%

### Iris Lite / Exp-B

- Upstream landmark frontend: `Iris-2.0-lite`
- Muse training: Multi-Stride + speed augmentation + start jitter
- Stride probabilities: s3=0.15, s4=0.40, s5=0.35, s6=0.10
- Training speed factor: 0.85-1.18, probability 0.70
- Start jitter: +/-3 frames
- Validation accuracy / Macro-F1: 98.380% / 98.357%
- Robustness-grid mean accuracy / Macro-F1: 94.583% / 94.507%
- Worst-cell accuracy / Macro-F1: 91.667% / 91.628%

## 3. Model architecture

Both ONNX files use the same A1-friendly Muse classifier architecture. The
models differ in the upstream landmark features and temporal training policy.

1. 1x1 stem convolution: 4 -> 64 channels
2. Four depthwise-separable stages: 64 -> 128 -> 192 -> 256 channels
3. Eight depthwise-separable blocks in total
4. Five same-channel residual additions
5. Three 2x2 max-pooling operations
6. Global average pooling
7. 1x1 classifier convolution: 256 -> 6 logits

Batch-normalization parameters used during training are folded into convolution
weights during ONNX export. The deployment graph contains no independent
BatchNormalization, Softmax, or Transpose node.

## 4. Dataset and split protocol

- 3 subjects
- 6 classes
- 40 original takes per subject/class
- 720 original takes and 198,720 grayscale TIFF frames
- Train/validation/test split: 504 / 108 / 108 original takes
- Split is stratified by subject x class
- All stride views from one take remain in the same split
- Leakage audit: train/validation/test take intersections are empty

Stride 5 and stride 6 contain boundary padding because an original take has 276
frames. They are treated as temporal augmentation/stress views, not as 64 fully
independent captured frames.

## 5. Common training configuration

- Epochs: 80
- Batch size: 64
- Optimizer: AdamW
- Initial learning rate: 1e-3
- Weight decay: 1e-4
- Label smoothing: 0.04
- Gradient clipping: max norm 1.0
- Learning-rate schedule: cosine annealing
- Basic augmentation: valid-point x/y Gaussian noise and temporal frame drop
- Best-model criterion: validation Macro-F1

## 6. Robustness protocol

Each final model is evaluated on the same 108 independent test takes under:

- source strides: 3, 4, 5, 6
- temporal speed factors: 0.75, 0.875, 1.0, 1.125, 1.25
- 20 test cells per model

The package reports both mean performance and worst-cell performance, rather
than relying on a single canonical-stride accuracy.

## 7. Package scope

Included:

- final ONNX models;
- model definition and final training/evaluation scripts;
- class mapping;
- best checkpoints when available;
- compact training and robustness reports;
- model structure summary, environment versions and SHA-256 hashes.

Excluded intentionally:

- raw images and extracted feature arrays;
- unpacking, sharding and bulk feature-extraction scripts;
- intermediate datasets and calibration samples;
- obsolete checkpoints and exploratory scripts.
