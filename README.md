# MonReader — Page Flip Detection

> A Computer Vision and Deep Learning project for detecting page-flipping actions from image sequences, powering the automated mobile document digitization system MonReader.

---

## Table of Contents

- [Overview](#overview)
- [About MonReader](#about-monreader)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach & Methodology](#approach--methodology)
  - [Approach 1: Custom CNN (Single Image)](#approach-1-custom-cnn-single-image)
  - [Approach 2: CNN + LSTM (Sequence Classification)](#approach-2-cnn--lstm-sequence-classification)
- [Results Summary](#results-summary)
- [Key Insights](#key-insights)
- [Recommendations](#recommendations)


---

## Overview

This project develops and evaluates deep learning models to detect whether a page is being flipped in smartphone camera footage — the critical first step in MonReader's fully automated document scanning pipeline. Two approaches are explored: a **Custom CNN** for single-image classification, and a **CNN + LSTM** hybrid for sequence-based temporal classification.

---

## About MonReader

MonReader is a mobile document digitization experience designed for the blind, researchers, and anyone who needs fast, fully automatic, high-quality document scanning in bulk.

The complete MonReader pipeline:

1. **Detects page flips** from low-resolution camera preview ← *this project*
2. **Captures a high-resolution photo** at the right moment
3. **Recognizes document corners** and crops accordingly
4. **Sharpens contrast** between text and background
6. **Recognizes text** with formatting intact via ML-powered redactor

---

## Problem Statement

| | |
|---|---|
| **Primary Goal** | Predict whether a page is being flipped using a **single image** |
| **Extended Challenge** | Predict whether a **sequence of images** contains a page flip |
| **Task Type** | Binary Classification: `flip` vs `notflip` |
| **Success Metric** | **F1 Score** (higher is better) |

---

## Dataset

Page-flipping videos were collected from smartphones and manually labeled as `flipping` or `not flipping`. Videos were clipped into short segments and individual frames extracted to disk.

### Dataset Statistics

| Split | `notflip` | `flip` | Total |
|-------|-----------|--------|-------|
| Training | 1,230 | 1,162 | **2,392** |
| Testing | 307 | 290 | **597** |

**Original image dimensions:** 1080 × 1920 px (RGB)

**Preprocessing:**
- Resized to `224 × 224` for the Custom CNN
- Resized to `96 × 96` per frame for CNN + LSTM sequences
- Pixel values normalized to `[0, 1]`
- Split into training, validation, and testing sets

---

## Approach & Methodology

### Approach 1: Custom CNN (Single Image)

A Sequential CNN was built from scratch to classify individual frames.

**Architecture:**
```
Input (224×224×3)
→ Conv2D (64 filters, 3×3, ReLU)
→ MaxPooling2D
→ BatchNormalization
→ Dropout
→ Conv2D (64 filters, 3×3, ReLU)
→ MaxPooling2D
→ BatchNormalization
→ Dropout
→ Flatten
→ Dense (64 units, ReLU)
→ Dense (1 unit, Sigmoid)
```

**Compiler:** Adam optimizer, Binary Crossentropy loss

**Initial Results:**

| Metric | Score |
|--------|-------|
| Test Accuracy | **95.31%** |
| F1 Score | **0.9510** |

**Saliency Map Analysis:**

Despite the high accuracy, saliency maps revealed a critical problem — the model was not learning page motion at all. Instead, it learned a **spurious correlation**: it associated the *presence of hands/wrists* with the `notflip` class as a shortcut, rather than focusing on page edges or motion blur. This is a classic example of a model achieving high accuracy for the wrong reasons.
<img width="1048" height="330" alt="image" src="https://github.com/user-attachments/assets/65354722-447e-462f-b49b-a39eee767710" />


**Impact of Data Augmentation:**

When retrained with data augmentation to break the spurious correlation, performance collapsed to ~50% accuracy (random chance). This confirmed the model's over-reliance on the hand-detection shortcut and inability to learn robust features without it.

**Conclusion:** High accuracy masked a fundamental flaw. Interpretability was essential to catching it.

---

### Approach 2: CNN + LSTM (Sequence Classification)

To address the shortcomings of the static CNN, a temporal model was developed that processes **sequences of 3 consecutive frames** to learn motion over time.

**Why this architecture?**

- A page flip is defined by *motion across time*, not any single static frame
- A CNN alone cannot understand temporal change
- The `TimeDistributed` wrapper applies the same CNN to every frame in a sequence, producing a series of feature vectors
- The `LSTM` then learns how those features evolve across frames, capturing the dynamics of a flip

**Architecture:**
```
Input: Sequence of 3 frames (96×96×3 each)
→ TimeDistributed(MobileNetV2)   ← pre-trained, frozen feature extractor
→ LSTM layer
→ Dropout
→ Dense (1 unit, Sigmoid)
```

**Key design choices:**
- **MobileNetV2** used as the base CNN for its efficiency and strong pre-trained visual features (`trainable=False`)
- **Sequence length = 3** frames per input
- **LSTM** captures temporal dependencies between consecutive frames

**Results:**

| Metric | Score |
|--------|-------|
| Test Accuracy | **99.73%** |
| F1 Score | **0.9970** |

**Saliency Map Analysis:**
Saliency maps for this model are expected to reflect genuine motion cues (page edges, blur, content transitions) rather than static hand presence, consistent with its dramatically higher performance and robustness.
<img width="1058" height="334" alt="image" src="https://github.com/user-attachments/assets/34821baa-e8f8-4ac9-b532-51de30634386" />


---

## Results Summary

| Model | Accuracy | F1 Score | Flaw |
|-------|----------|----------|------|
| Custom CNN (baseline) | 95.31% | 0.9510 | Spurious correlation (hand detection) |
| Custom CNN + Augmentation | ~50% | — | Collapsed without shortcut |
| **CNN + LSTM (MobileNetV2)** | **99.73%** | **0.9970** | Requires sequential frame input |

---

## Key Insights

1. **Page flipping depends on motion over time.** Page flipping is a motion event. Models that process single static frames are fundamentally limited and prone to learning shortcuts rather than actual page dynamics.

2. **High accuracy does not always mean the model is learning correctly.** The Custom CNN's 95% accuracy looked great on paper, but saliency maps revealed it was useless for the real task. High accuracy alone is not sufficient validation.

3. **Models can learn the wrong patterns from data.** The CNN learned that "hand present = not flipping" — a pattern that would completely break down in real-world conditions where hands appear in both classes.

4. **Transfer learning helped improve feature extraction.** Using a pre-trained MobileNetV2 as a frozen feature extractor gave the LSTM high-quality visual representations from the start, without needing to learn low-level features from scratch on a small dataset.

5. **Data augmentation alone cannot fix a weak learning strategy.** Adding augmentation without first addressing the root cause of spurious learning can confuse a model that depends on a shortcut, rather than teaching it the right features.

---

## Recommendations

1. **Saliency maps for CNN+LSTM:** Perform detailed saliency analysis on the sequence model to visually confirm it focuses on page edges, motion blur, and inter-frame content change — not hands.

2. **Misclassification analysis:** Inspect the small number of misclassified sequences (false positives and negatives) to identify edge cases, labeling issues, or weak spots in the model's temporal understanding.

3. **Real-world testing:** Deploy the CNN+LSTM in a simulated MonReader environment across diverse lighting conditions, hand positions, and page/book types.

---

