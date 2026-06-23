# MonReader — Page Flip Detection

> A Computer Vision and Deep Learning project to detect page-flipping actions from images, powering the automated mobile document digitization experience **MonReader**.

---

## Table of Contents

- [Overview](#overview)
- [About MonReader](#about-monreader)
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Approach: Custom CNN](#approach-custom-cnn)
- [The Spurious Correlation Problem](#the-spurious-correlation-problem)
- [Failure of Augmentation](#failure-of-augmentation)
- [Key Insights](#key-insights)
- [Recommendations & Next Steps](#recommendations--next-steps)


---

## Overview

This project focuses on **Page-Flip Detection** for MonReader, a mobile document digitization experience designed for the blind, researchers, and anyone who needs fast, fully automatic, high-quality document scanning. The core idea is to automatically detect page flips from a low-resolution camera preview, triggering high-resolution image capture and the downstream processing pipeline (cropping, dewarping, text recognition, and ML-powered redaction).

---

## About MonReader

MonReader handles the entire document scanning experience — the user simply flips pages, and the app does the rest:

1. **Detects page flips** from low-resolution camera preview ← *this project*
2. **Captures a high-resolution photo** at the right moment
3. **Recognizes document corners** and crops accordingly
4. **Dewarps** the cropped document into a bird's-eye view
5. **Sharpens contrast** between text and background
6. **Recognizes text** with formatting intact, refined by an ML-powered redactor

---

## Project Goal

**To predict if a page is being flipped using a single image.**

| | |
|---|---|
| **Task Type** | Binary Classification: `flip` vs `notflip` |
| **Input** | A single RGB image |
| **Success Metric** | F1 Score (higher is better) |

---

## Dataset

The dataset consists of images extracted from page-flipping videos, labeled as `flip` or `notflip`.

- **Source:** Downloaded from Google Drive as `monreader_data.zip` and extracted
- **Original image size:** 1080 × 1920 px (RGB)
- **Preprocessing:** Resized to `224 × 224`, normalized to pixel values in `[0, 1]`
- **Split:** Training, validation, and testing sets, with labels reshaped for model compatibility

### Dataset Statistics

| Split | `notflip` | `flip` | Total |
|-------|-----------|--------|-------|
| Training | 1,230 | 1,162 | **2,392** |
| Testing | 307 | 290 | **597** |
<img width="1051" height="484" alt="image" src="https://github.com/user-attachments/assets/29ec75a7-a47a-4c12-8983-02b7b26f818a" />
---

## Approach: Custom CNN

A Sequential CNN was built from scratch to classify individual frames as `flip` or `notflip`.

**Architecture:**
```
Input (224×224×3)
→ Conv2D (64 filters, 3×3, ReLU)
→ MaxPooling2D
→ BatchNormalization
→ Dropout
→ Conv2D (64 filters, 3×3, ReLU)
→ MaxPooling2D
→ Flatten
→ Dense (64 units, ReLU)
→ Dense (1 unit, Sigmoid)
```

**Compiler:** Adam optimizer, Binary Crossentropy loss

**Results:**

| Metric | Score |
|--------|-------|
| Test Accuracy | **~0.99** |
| F1 Score | **~0.99** |

Training converged well over 5 epochs, with training and validation performance closely tracked — on the surface, an excellent result.

---

## The Spurious Correlation Problem

Despite the near-perfect accuracy, **saliency map analysis** revealed a critical flaw in the model's reasoning.

> The model was primarily focusing on the presence of **hands and wrists** — particularly in the `notflip` class — rather than understanding the actual visual cues of a page flip in motion.

This is a spurious correlation, not genuine motion understanding. The model was likely overfitting to an artifact in the dataset rather than learning generalizable, page-motion-related features — meaning the headline accuracy score was misleading about true real-world readiness.
<img width="813" height="538" alt="image" src="https://github.com/user-attachments/assets/e1b0b31f-929f-4677-a733-b17e312cfa81" />

---

## Failure of Augmentation

The Custom CNN was retrained with additional **BatchNormalization** and **Dropout** layers, using adjusted, less aggressive data augmentation (rotation, shift, brightness, zoom, shear).

**Result:** The model failed to learn effectively, performing **no better than random chance**.

This failure suggests one of two things:
- The model architecture isn't robust enough to handle the increased variability introduced by augmentation, **or**
- The dataset — even with augmentation — is insufficient for the model to learn the complex, non-spurious features required for true motion detection

---

## Key Insights

1. **High accuracy can be misleading.** A ~99% accuracy/F1 score initially looked like a success, but interpretability tools exposed that the model was solving the wrong problem.

2. **Interpretability is essential, not optional.** Without saliency map analysis, this spurious correlation would have gone undetected until real-world deployment failure.

3. **Augmentation alone doesn't fix shortcut learning.** Simply adding more visual variability without addressing the root cause of the spurious correlation caused the model to collapse rather than improve.

4. **Dataset limitations may be at play.** The dataset, while labeled, may contain biases — such as a strong association between visible hands and a particular class — that make it easy for a model to latch onto misleading patterns instead of true motion cues.

5. **A single image may be fundamentally limited.** Page flipping is a temporal, motion-based event. A model relying on one static frame may always be vulnerable to learning superficial visual shortcuts instead of real flip dynamics.

---

## Recommendations & Next Steps

### 1. Refine the Data Augmentation Strategy
- **Motion-specific augmentation:** Simulate flipping motions, blur, and varying speeds (e.g., motion blur, optical flow, synthetic data generation) rather than relying only on geometric transforms.
- **Contextual augmentation:** Vary hand positions, book types, lighting, and backgrounds to prevent the model from latching onto simple hand detection.
- **Targeted augmentation:** Apply different augmentation strategies per class if specific weaknesses are identified.

### 2. Focus on True Motion Features
- **Temporal information:** Incorporate sequences of images (e.g., via LSTMs or 3D CNNs) to capture frame-to-frame change directly.
- **Optical flow:** Extract optical flow features as explicit motion vectors between frames, giving the model direct access to motion information.

### 3. Advanced Interpretability & Debugging
- Continue using **Grad-CAM** and explore **LIME** to monitor what the model focuses on across new architectures and augmentations.
- Generate **adversarial examples** to probe and identify features the model incorrectly relies upon.

### 5. Dataset Expansion & Curation
- **Diversity:** Source or generate data that explicitly breaks the spurious correlation — e.g., `notflip` images without hands, `flip` images with varied hand placements.
- **Annotation refinement:** Review existing labels for ambiguity, especially around borderline flip/notflip cases.

> **Ultimate goal:** Build a page-flip detector that is not just accurate, but **robust** — one that genuinely learns the physical act of page-flipping rather than relying on superficial visual cues.

---


---

*Built with Python · TensorFlow · Computer Vision · Interpretable AI*
