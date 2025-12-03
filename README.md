# 🛡️ E-Commerce Fraud Detection - Multiclass Classification

A comprehensive machine learning project for detecting fraudulent e-commerce transactions using multiclass classification.

[![Julia](https://img.shields.io/badge/Julia-1.x-9558B2?logo=julia)](https://julialang.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Fraud%20Detection-blue)](https://github.com/Fabrimagic1/progetto-machine-Learning)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Approaches](#approaches)
- [Algorithms](#algorithms)
- [Results](#results)
- [Team](#team)

## 🎯 Overview

This project implements a **3-class fraud detection system** for e-commerce transactions:

- **Class 0 (LEGITTIMO)**: Low-risk, legitimate transactions
- **Class 1 (SOSPETTO)**: Suspicious transactions requiring manual review
- **Class 2 (FRAUDOLENTO)**: High-risk fraudulent transactions

This graduated risk assessment enables businesses to:
- ✅ Automatically approve low-risk transactions
- ⚠️ Flag suspicious cases for manual review
- ❌ Immediately block high-risk frauds

## 📊 Dataset

**Source**: Fraudulent E-Commerce Transaction Data (1.5M transactions)

**Original Format**: Binary fraud labels (fraud vs non-fraud)

**Our Enhancement**: 3-class risk assessment based on multiple signals:
- ⏰ **Time Risk**: Night transactions (0-5am, 11pm)
- 💰 **Amount Risk**: High-value transactions (>90th percentile)
- 👤 **Account Age Risk**: New accounts (<30 days)

**Class Distribution** (after balancing):
- Legitimate: 33.3%
- Suspicious: 33.3%
- Fraudulent: 33.3%

## 🔬 Methodology

### Three Experimental Approaches

#### **Approach 1: Standard 80/20 Split**
- Training: 80% (49,569 samples)
- Testing: 20% (12,393 samples)
- All features used

#### **Approach 2: Balanced 50/50 Split**
- Training: 50% (30,981 samples)
- Testing: 50% (30,981 samples)
- Tests generalization with less training data

#### **Approach 3: PCA Dimensionality Reduction**
- Principal Component Analysis applied
- 80/20 split with reduced features
- Tests if dimensionality reduction improves performance

### Validation Strategy

- **3-Fold Stratified Cross-Validation** on training set
- **Hold-out Test Set** for final evaluation (never used during training)
- **Random Seed**: 42 (for reproducibility)

## 📁 Project Structure

```
progetto-machine-Learning/
├── main.jl                           # Main executable script
├── utils/
│   ├── utils.jl                     # Course utilities (modelCrossValidation, etc.)
│   └── preprocessing.jl             # Custom preprocessing functions
├── datasets/
│   └── Fraudulent_E-Commerce_Transaction_Data_merge.csv
└── README.md                         # This file
```

## 🔧 Requirements

### Julia Packages

```julia
using CSV
using DataFrames
using Statistics
using Dates
using StatsBase
using LinearAlgebra
using Random
using MLJ
using Flux
```

### ML Libraries

- **LIBSVM**: Support Vector Machines
- **DecisionTree**: Decision Tree Classifier
- **NearestNeighborModels**: k-NN Classifier
- **Flux**: Neural Networks

## 💻 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Fabrimagic1/progetto-machine-Learning.git
cd progetto-machine-Learning
```

2. **Install Julia** (1.x or higher)
   - Download from [julialang.org](https://julialang.org/downloads/)

3. **Install dependencies**
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Statistics", "Dates", "StatsBase", 
         "LinearAlgebra", "Random", "MLJ", "Flux"])
```

4. **Load ML models**
```julia
using MLJ
@load SVC pkg=LIBSVM
@load DecisionTreeClassifier pkg=DecisionTree
@load KNNClassifier pkg=NearestNeighborModels
```

## 🚀 Usage

### Run Complete Pipeline

```bash
julia main.jl
```

The script is designed to be **executable from top to bottom** and will:
1. Load and preprocess the dataset
2. Create 3-class target variable
3. Balance classes
4. Run all three approaches
5. Test 4 ML algorithms with multiple configurations each
6. Evaluate ensemble methods
7. Display comprehensive results

### Expected Runtime

- **Approach 1 (80/20)**: ~30-45 minutes
- **Approach 2 (50/50)**: ~20-30 minutes
- **Approach 3 (PCA)**: ~25-35 minutes
- **Total**: ~1.5-2 hours

## 📚 Approaches

### Approach 1: Standard 80/20 Split

**Purpose**: Baseline performance with standard train/test split

**Configuration**:
- Training: 80% of balanced dataset
- Testing: 20% held out for final evaluation
- All 8 engineered features used

**Best Results**:
- ANN: F1 ~XX%
- Decision Tree: F1 ~XX%
- kNN: F1 ~XX%

### Approach 2: Balanced 50/50 Split

**Purpose**: Test generalization with less training data

**Configuration**:
- Training: 50% of balanced dataset
- Testing: 50% held out for final evaluation
- Tests model robustness with limited data

**Key Insight**: Evaluates how much training data is really needed

### Approach 3: PCA Dimensionality Reduction

**Purpose**: Test if dimensionality reduction improves performance

**Configuration**:
- PCA applied to reduce features
- 80/20 split
- Optimal number of components selected via explained variance

**Key Insight**: Can simpler representations improve generalization?

## 🤖 Algorithms

### 1. Artificial Neural Networks (ANNs)
**8 Topologies Tested**:
- 1 hidden layer: [256], [128], [64], [32]
- 2 hidden layers: [256,128], [128,64], [64,32], [96,48]

**Configuration**:
- Activation: Sigmoid (hidden), Softmax (output)
- Optimizer: Adam (learning rate: 0.003)
- Loss: Cross-entropy
- Early stopping: Patience 25 epochs

### 2. Support Vector Machines (SVMs)
**8 Configurations Tested**:
- Linear kernel: C = 1.0, 10.0
- RBF kernel: C = 0.1, 1.0, 10.0
- Polynomial kernel: degree 2 & 3, various C values

### 3. Decision Trees
**6 Maximum Depths Tested**:
- Depths: 3, 5, 7, 10, 15, Unlimited
- Criterion: Gini impurity
- Min samples split: 2

### 4. k-Nearest Neighbors (kNN)
**6 k Values Tested**:
- k = 1, 3, 5, 7, 10, 15
- Distance: Euclidean
- Voting: Majority

### 5. Ensemble Methods
**Two Strategies**:
1. **Majority Voting**: Each model votes equally
2. **Weighted Voting**: Models vote proportionally to CV F1 scores

**Models Combined**: ANN + Decision Tree + kNN

## 📈 Results

### Approach 1: 80/20 Split

| Rank | Approach | F1 Score | Accuracy |
|------|----------|----------|----------|
| 1 | ANN (best topology) | XX.XX% | XX.XX% |
| 2 | Decision Tree | XX.XX% | XX.XX% |
| 3 | kNN | XX.XX% | XX.XX% |
| 4 | SVM | XX.XX% | XX.XX% |
| 5 | Ensemble (Majority) | XX.XX% | XX.XX% |
| 6 | Ensemble (Weighted) | XX.XX% | XX.XX% |

### Key Findings

1. **Best Individual Model**: [To be filled after running experiments]
2. **Ensemble Performance**: [Comparison with individual models]
3. **Computational Efficiency**: [Runtime comparisons]
4. **Generalization**: [50/50 vs 80/20 comparison]
5. **Dimensionality Impact**: [PCA results]

## 🎓 Team

**Course**: Machine Learning I  
**Project Type**: Group Project  
**Institution**: [Your University]

## 📝 Features Engineering

**Final Feature Set (8 features)**:
1. Transaction Amount
2. Account Age Days
3. Transaction Hour
4. Is Night (binary flag)
5. Amount per Account Age
6. High Value Flag (>95th percentile)
7. New Account Flag (<30 days)
8. [Additional engineered feature]

**Preprocessing Pipeline**:
1. Time feature extraction
2. Feature engineering
3. Missing value imputation (median)
4. Feature selection (numerical only)
5. Min-Max normalization [0,1]

## 🔍 Model Selection

**Cross-Validation**:
- 3-fold stratified CV on training set
- Stratification ensures balanced class distribution in each fold

**Hyperparameter Selection**:
- Multiple configurations tested per algorithm
- Best configuration selected based on CV F1 score

**Final Evaluation**:
- Selected model retrained on full training set
- Evaluated on hold-out test set (never seen during training)

## ⚡ Performance Metrics

**Primary Metric**: F1 Score (harmonic mean of precision and recall)

**Additional Metrics**:
- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision (PPV)
- Negative Predictive Value (NPV)
- Per-class metrics

## 🛠️ Reproducibility

**Random Seed**: 42 (set at beginning of script)

**Deterministic Results**:
- All random operations seeded
- MLJ models use consistent RNG
- Cross-validation indices fixed

## 📖 References

- Dataset: Fraudulent E-Commerce Transaction Data
- Course Materials: Machine Learning I utilities
- Libraries: MLJ.jl, Flux.jl, LIBSVM.jl

## 📄 License

Academic project - [Your License]

## 🤝 Contributing

This is an academic project. Contributions are not currently accepted.

## 📧 Contact

For questions about this project, please contact: [Your Email]

---

**⭐ If you find this project useful, please consider giving it a star!**
