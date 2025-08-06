# H&M Personalized Fashion Recommendations

**Kaggle Competition Solution** - [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview)

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-orange.svg)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
[![Rank](https://img.shields.io/badge/Rank-45%2F3006-brightgreen.svg)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/leaderboard)
[![Score](https://img.shields.io/badge/Score-0.0292-yellow.svg)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/leaderboard)

![background](./imgs/img1.png)

![rank](./imgs/img2.png)

## 📋 Table of Contents

- [🏆 Competition Results](#-competition-results)
- [🎯 Solution Overview](#-solution-overview)
- [🚀 Quick Start](#-quick-start)
- [📥 Pre-trained Embeddings](#-pre-trained-embeddings)
- [📁 Project Structure](#-project-structure)
- [🔧 Technical Architecture](#-technical-architecture)
- [👥 About the Team](#-about-the-team)

## 🎯 Quick Summary

| Metric | Value |
|--------|-------|
| **Competition Rank** | 45th out of 3,006 teams |
| **Public Score** | 0.0292 |
| **Private Score** | 0.02996 |
| **Improvement** | +0.0006 over single strategy |
| **Architecture** | Two-stage retrieval + ranking |
| **Models** | LightGBM + DNN ensemble |
| **Memory** | Optimized for 50GB RAM |

## 🏆 Competition Results

This repository contains our **final solution** that achieved **45th place out of 3,006 teams** in the H&M Personalized Fashion Recommendations competition.

- **Public Leaderboard Score**: 0.0292
- **Private Leaderboard Score**: 0.02996
- **Team Ranking**: 45/3006

## 🎯 Solution Overview

Our solution implements a **two-stage recommendation system** with ensemble methods:

### Stage 1: Retrieval (Candidate Generation)
- **Two distinct recall strategies** to ensure candidate diversity
- Multiple retrieval rules: collaborative filtering, item-based, popularity-based, and temporal patterns
- Quality control mechanisms to maintain positive rates

### Stage 2: Ranking (Model Training)
- **Three different ranking models** for each strategy:
  - LightGBM Ranker
  - LightGBM Classifier  
  - Deep Neural Network (DNN)
- **Ensemble blending** of all models for final predictions

### Key Innovation: Ensemble Diversity
The two recall strategies generate quite different candidates, enabling effective ensemble methods:
- Single strategy performance: 0.0286
- **Ensemble performance: 0.0292** (+0.0006 improvement)
- Enhanced robustness through model diversity

### Hardware Optimization
Due to 50GB RAM constraints, we optimized for efficiency:
- Average 50 candidates per user
- 4 weeks of training data
- Memory-efficient data processing pipeline

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- 50GB+ RAM (for full dataset processing)
- Jupyter Notebook

### Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/ABHIRAM1234/H-M-Fashion-Recommendations.git
   cd H-M-Fashion-Recommendations
   ```

2. **Set up environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up data structure**
   - Create the data directory structure as shown below
   - Download the Kaggle competition dataset files and place them in `data/raw/`:
     - `articles.csv`
     - `customers.csv` 
     - `transactions_train.csv`
     - `sample_submission.csv`

4. **Download pre-trained embeddings**
   - Option A: Generate embeddings using [Embeddings.ipynb](notebooks/Embeddings.ipynb)
   - Option B: Download pre-trained embeddings from the links below and place them in `data/external/`

5. **Run the pipeline**
   - Start with `LGB Recall 1.ipynb` (generates features used by all models)
   - Then run the remaining notebooks in sequence

## 📥 Pre-trained Embeddings

Download these pre-trained embedding files and place them in `data/external/`:

### Collaborative Filtering Embeddings
- [dssm_item_embd.npy](https://drive.google.com/file/d/13rGRbevjcd0yZdwuOTPmNyMOIx9WOLb9/view?usp=sharing) - DSSM item embeddings
- [dssm_user_embd.npy](https://drive.google.com/file/d/13nkDc7Dt6QtXx91i3sjnotQNGX2JpSk_/view?usp=sharing) - DSSM user embeddings
- [yt_item_embd.npy](https://drive.google.com/file/d/11Q8nWxOlSTspQwH9OGmR9vGoAqJ2wWbS/view?usp=sharing) - YouTube-style item embeddings
- [yt_user_embd.npy](https://drive.google.com/file/d/11OX9vuHmCrCk8Mcl6XA1TF0l0nBL___j/view?usp=sharing) - YouTube-style user embeddings

### Word2Vec Embeddings
- [w2v_item_embd.npy](https://drive.google.com/file/d/1-8spKOVtb0jr5xYT8oMKMC5z3BPpCOU-/view?usp=sharing) - Word2Vec CBOW item embeddings
- [w2v_user_embd.npy](https://drive.google.com/file/d/1-6CAnA2_pHXrhCyplV-WsI9lreSf6Rm-/view?usp=sharing) - Word2Vec CBOW user embeddings
- [w2v_product_embd.npy](https://drive.google.com/file/d/1-R8Rww7QqHZOIcyIhZxEMiXRW1hzJ5wI/view?usp=sharing) - Word2Vec CBOW product embeddings
- [w2v_skipgram_item_embd.npy](https://drive.google.com/file/d/1-AmzbyCHx9i0CddZIdbqNJPAMXw3Kg34/view?usp=sharing) - Word2Vec Skip-gram item embeddings
- [w2v_skipgram_user_embd.npy](https://drive.google.com/file/d/1-8BpDQUn310Vns72t1up3uIOOnV_nR4h/view?usp=sharing) - Word2Vec Skip-gram user embeddings
- [w2v_skipgram_product_embd.npy](https://drive.google.com/file/d/1-QhHbFr16koCBL5OIMHxJX9ZAQJAhbHF/view?usp=sharing) - Word2Vec Skip-gram product embeddings

### Visual Embeddings
- [image_embd.npy](https://drive.google.com/file/d/1-WkIeInVvHJz4ScA3n-CRyVLQjW51gDH/view?usp=sharing) - Image-based embeddings

Project Organization
------------

    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── external       <- External data source, e.g. article/customer pre-trained embeddings.
    │   ├── interim        <- Intermediate data that has been transformed, e.g. Candidates generated form recall strategies.
    │   ├── processed      <- Processed data for training, e.g. dataframe that has been merged with generated features.
    │   └── raw            <- The original dataset.
    │
    ├── docs               <- Sphinx docstring documentation.
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to preprocess data
        │   ├── datahelper.py
        │   └── metrics.py
        │
        ├── features       <- Scripts of feature engineering
        │   └── base_features.py
        │
        └── retrieval      <- Scripts to generate candidate articles for ranking models
            ├── collector.py
            └── rules.py

## 🔧 Technical Architecture

### Core Components

#### 1. **Data Processing Pipeline** (`src/data/`)
- **DataHelper**: Handles data loading, ID encoding, feature engineering, and memory optimization
- **Metrics**: Implements MAP@k, Recall@k, and Hit Rate@k evaluation metrics

#### 2. **Feature Engineering** (`src/features/`)
- **Temporal Features**: Sales patterns, popularity trends, seasonal effects
- **Behavioral Features**: Purchase history, repurchase ratios, user preferences
- **Product Features**: Categories, colors, materials, gender classification

#### 3. **Retrieval System** (`src/retrieval/`)
- **RuleCollector**: Orchestrates multiple retrieval rules and quality control
- **Retrieval Rules**: Collaborative filtering (ALS, BPR), item-based CF, popularity-based, temporal patterns

#### 4. **Model Ensemble**
- **LightGBM**: Gradient boosting for ranking and classification tasks
- **Deep Neural Networks**: Multi-layer perceptrons for complex pattern learning
- **Ensemble Blending**: Weighted combination of multiple models

### Key Features

- **Memory Optimization**: Efficient data types and batch processing for large datasets
- **Quality Control**: Positive rate monitoring and adaptive candidate filtering
- **Modular Design**: Reusable components for different recommendation scenarios
- **Comprehensive Documentation**: Sphinx-generated API documentation

## 🛠️ Technologies & Skills Demonstrated

### **Machine Learning & AI**
- **Recommendation Systems**: Collaborative filtering, content-based filtering, hybrid approaches
- **Deep Learning**: Neural networks for ranking and embedding generation
- **Ensemble Methods**: Model blending and stacking techniques
- **Feature Engineering**: Temporal, behavioral, and product features

### **Data Engineering**
- **Large-Scale Processing**: 50GB+ dataset optimization
- **Memory Management**: Efficient data types and batch processing
- **Data Pipeline**: End-to-end ETL with quality control
- **Embedding Generation**: Word2Vec, DSSM, YouTube-style embeddings

### **Software Engineering**
- **Modular Architecture**: Clean separation of concerns
- **Documentation**: Comprehensive API docs with Sphinx
- **Testing**: Professional project structure with testing setup
- **Version Control**: Proper Git workflow and contribution guidelines

### **Tools & Frameworks**
- **Python**: pandas, numpy, scikit-learn, lightgbm
- **Deep Learning**: TensorFlow, Keras
- **Collaborative Filtering**: implicit library (ALS, BPR)
- **Documentation**: Sphinx, Jupyter notebooks
- **Development**: pytest, black, flake8

## 👥 About the Team

This project was developed as a comprehensive solution to the H&M Personalized Fashion Recommendations Kaggle competition by our team. The solution demonstrates advanced techniques in:

- **Recommendation Systems**: Two-stage retrieval and ranking architecture
- **Machine Learning**: Ensemble methods, collaborative filtering, and deep learning
- **Data Engineering**: Large-scale data processing and memory optimization
- **Software Engineering**: Modular design, comprehensive documentation, and production-ready code

The project showcases practical application of recommendation system theory to real-world e-commerce data, achieving top-tier performance in a competitive environment.

### Team Member
**Abhiram MV** - Lead Developer and Data Scientist
- GitHub: [@ABHIRAM1234](https://github.com/ABHIRAM1234)
- Repository: [H-M-Fashion-Recommendations](https://github.com/ABHIRAM1234/H-M-Fashion-Recommendations)

### Acknowledgments
This project is based on the original work by [Wp-Zhang](https://github.com/Wp-Zhang/H-M-Fashion-RecSys) and has been enhanced with comprehensive documentation, improved code structure, and professional project setup for educational and portfolio purposes.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
