# 📖 MonReader — Page Flip Detection

> A Computer Vision and Deep Learning project for detecting page-flipping actions from image sequences to support automated mobile document digitization.

---

## Table of Contents

- [Overview](#overview)
- [About MonReader](#about-monreader)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Success Metrics](#success-metrics)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Results](#results)

---

## Overview

This project tackles the core perception challenge behind **MonReader** — a mobile document digitization system. Specifically, it builds a deep learning model capable of determining whether a page is being flipped in a given image frame. Accurate page-flip detection is the trigger that allows MonReader to capture high-resolution document scans at exactly the right moment.

---

## About MonReader

MonReader is a mobile document digitization experience designed for the blind, researchers, and anyone who needs fast, fully automatic, high-quality document scanning in bulk.

The full MonReader pipeline:
1. **Detects page flips** from a low-resolution camera preview
2. **Captures a high-resolution photo** of the document at the right moment
3. **Recognizes document corners** and crops the image accordingly
4. **Dewarps** the cropped document to produce a bird's-eye view
5. **Sharpens contrast** between text and background
6. **Recognizes text** with formatting intact, refined by MonReader's ML-powered redactor

This repository focuses on **step 1** — the page flip detection model.

---

## Problem Statement

**Primary Goal:** Predict whether a page is being flipped using a single image frame.

**Extended Challenge:** Predict whether a given *sequence* of images contains a page-flipping action.

This is framed as a **binary classification** task:
- `flipping` — a page flip is in progress
- `not flipping` — no flip is occurring

---

## Dataset

Page-flipping videos were collected from smartphones and manually labeled as `flipping` or `not flipping`. Each video was clipped into short segments, and individual frames were extracted and saved to disk.

**Frame naming convention:**
```
VideoID_FrameNumber
```
For example: `video001_0042.jpg`

**Download the dataset:**
[Google Drive — MonReader Dataset](https://drive.google.com/file/d/1KDQBTbo5deKGCdVV_xIujscn5ImxW4dm/view?usp=sharing)

---

## Project Structure

```
MonReader-PageFlip-Detection/
│
├── MonReader.ipynb       # Main notebook: EDA, modeling, and evaluation
└── README.md
```

---

## Methodology

The notebook (`MonReader.ipynb`) walks through the full machine learning pipeline:

1. **Exploratory Data Analysis (EDA)** — visualizing frame samples, class distribution, and data characteristics
2. **Data Preprocessing** — loading frames, resizing, normalization, and train/validation splitting
3. **Model Architecture** — a Convolutional Neural Network (CNN) built with deep learning frameworks (e.g., TensorFlow/Keras or PyTorch) for binary image classification
4. **Training** — model training with appropriate callbacks (early stopping, learning rate scheduling)
5. **Evaluation** — performance assessed via F1 score on the validation/test set
6. **Inference** — predicting flip/no-flip on individual frames or sequences

---

## Success Metrics

| Metric | Goal |
|--------|------|
| **F1 Score** | Maximize (higher is better) |

F1 score is used as the primary metric because it balances precision and recall — important when both false positives (unnecessary captures) and false negatives (missed page flips) carry real costs.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rinalpatel21/MonReader-PageFlip-Detection.git
cd MonReader-PageFlip-Detection
```

### 2. Download the dataset

Download the dataset from the link above and place the extracted frames in a local `data/` directory:

```
data/
├── flipping/
│   ├── video001_0001.jpg
│   └── ...
└── not_flipping/
    ├── video002_0001.jpg
    └── ...
```

### 3. Launch the notebook

```bash
jupyter notebook MonReader.ipynb
```

---

## Requirements

```
python >= 3.8
jupyter
numpy
pandas
matplotlib
scikit-learn
tensorflow >= 2.x   # or pytorch
opencv-python
Pillow
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python Pillow jupyter
```

---

## Results

> Model performance results and visualizations (confusion matrix, training curves, sample predictions) are available inside `MonReader.ipynb`.

---

## Acknowledgements

This project is built around the MonReader product concept, a mobile document scanning solution designed to make document digitization accessible — particularly for the visually impaired.

---

*Built with Python · Deep Learning · Computer Vision*
