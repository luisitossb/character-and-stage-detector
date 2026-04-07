# MobileNet-based Character & Stage Classifier

A real-time image classifier that detects Smash Bros Ultimate characters and stages from in-game screenshots using PyTorch and MobileNetV3.

## Overview

This project uses transfer learning with a pre-trained MobileNetV3-Small model to classify:
- **Characters**: All 89+ fighters with character variants (e.g., Bowser Jr alts, Hero alts)
- **Stages**: All competitive Smash Bros stages

The classifier runs in real-time on live gameplay footage, capturing screenshots, processing them through trained models, and outputting detected matchups with confidence scores.

## Features

- **Transfer Learning**: Fine-tunes MobileNetV3-Small on labeled character sprite data
- **Real-Time Inference**: Processes game screenshots at interactive speeds
- **Voting System**: Implements majority voting across frame windows to reduce noise and improve accuracy
- **Data Augmentation**: ColorJitter, random flips, and rotation for robust model training
- **Modular Design**: Separate modules for character detection, stage detection, screen capture, and logging

## Architecture

- **Character Classifier**: MobileNetV3-Small fine-tuned on 89+ character classes with variant support
- **Stage Classifier**: Separate MobileNetV3-Small model for stage detection
- **Screen Capture**: Captures game footage and crops relevant regions (P1/P2 portraits)
- **Voting Pipeline**: Accumulates predictions across 5 frames, confirms with 3+ votes for stability

## Model Training

Both character and stage classifiers are trained using:
- **Base Model**: MobileNetV3-Small (ImageNet pretrained weights)
- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: Cosine Annealing (50 epochs)
- **Loss**: Cross-Entropy Loss
- **Batch Size**: 32
- **Data Split**: 80% train, 20% validation

### Data Processing

- Character images: Named folders with format `[ID] - [Character Name]`
- Sprite variants: Indexed as `chara_0_[name]_[alt].png` (alts 00-07)
- Preprocessing: Resize to 128×128, handle alpha channels, normalize to ImageNet statistics

## Usage

### Training

Train character model:
```bash
python train_character_model.py
```

Train stage model:
```bash
python train_stage_model.py
```

### Inference

Run real-time detection:
```bash
python main.py
```

The classifier will:
1. Capture game screenshots
2. Identify character portraits for P1 and P2
3. Identify the active stage
4. Apply voting logic for stability
5. Output detected matchup when confirmed

## Files

- `main.py`: Real-time inference pipeline with voting system
- `train_character_model.py`: Character model training script
- `train_stage_model.py`: Stage model training script
- `character_check.py`: Character inference and template loading
- `stage_check.py`: Stage inference and index management
- `screen_capture.py`: Game footage capture and cropping
- `config.py`: Configuration (regions, poll interval, model paths)
- `write_out.py`: Output handler for detected matchups
- `build_character_index.py`: Utility to index character classes
- `build_stage_index.py`: Utility to index stage classes

## Dependencies

- PyTorch
- TorchVision
- OpenCV (cv2)
- NumPy

## Notes

- Models are saved to `data/character_model.pth` and `data/stage_model.pth`
- Class indices are cached in `data/character_classes.npy` and `data/stage_classes.npy`
- Accuracy improves significantly with larger, more diverse training datasets
- Current implementation optimized for Smash Bros Ultimate character portraits

## Future Improvements

- Expand training data to improve classification accuracy across lighting conditions
- Implement character variant detection (e.g., costume detection)
- Add confidence thresholding and rejection sampling
- Optimize inference speed for lower-end hardware
- Support for match state detection (menu, gameplay, results screen)
