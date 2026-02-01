# Synthetic Text Detection: Human vs AI Movie Reviews

## Overview
This project focuses on **detecting AI-generated text versus human-written text** in the context of movie reviews. With the rapid adoption of Large Language Models (LLMs), distinguishing organic human expression from synthetic content has become a critical challenge.

The project evaluates and compares **traditional machine learning**, **ensemble learning**, and **deep learning** approaches, highlighting an important trade-off between **numerical accuracy** and **real-world robustness** against “humanized” AI text.

---

## Problem Statement
Given a movie review, classify whether the text is:
- **Human-written (Label 0)**
- **AI-generated (Label 1)**

The challenge is amplified by *adversarial AI text*, where models attempt to mimic casual human tone, slang, and emotional phrasing.

---

## Dataset
- **Total Samples:** 150  
- **Class Distribution:**  
  - 75 Human-written reviews (IMDb)  
  - 75 AI-generated reviews (Gemini)

### AI Class Variants
- **Standard AI:** Structured, summary-style text  
- **Humanized AI:** Prompt-engineered AI text designed to mimic informal human writing  

The dataset is **strictly balanced** to avoid class bias.

---

## Feature Engineering

### 1. TF-IDF (Term Frequency–Inverse Document Frequency)
Used for all classical machine learning and ensemble models.

- Converts text into numerical vectors based on word importance
- Highlights discriminative keywords
- Filters out common stop words
- Effective for lexical and frequency-based patterns

### 2. BERT Embeddings
Used for deep learning and transformer-based models.

- Captures **contextual and semantic meaning**
- Considers word order and sentence structure
- More robust against stylistic manipulation

---

## Model Architectures

### Traditional & Ensemble Methods
- **Logistic Regression (Baseline)**  
  Linear probabilistic classifier using TF-IDF features.

- **Random Forest (Bagging)**  
  Multiple decision trees trained on bootstrap samples to reduce variance.

- **XGBoost (Boosting)**  
  Sequential tree-based model focusing on hard-to-classify samples.

- **Stacking Classifier**  
  - **Base Models:** Naive Bayes + Random Forest  
  - **Meta Model:** Logistic Regression  
  Combines heterogeneous learners to improve stability and accuracy.

### Deep Learning Methods
- **BERT (Transfer Learning)**  
  Uses `bert-base-uncased` to extract contextual embeddings, followed by classification.

- **LSTM (RNN)**  
  Sequence-based neural network trained from scratch to model word order and long-term dependencies.

---

## Quantitative Results (Validation Accuracy)

| Model | Accuracy |
|------|---------|
| Logistic Regression (TF-IDF) | ~86.7% |
| Random Forest (Bagging) | ~83.3% |
| XGBoost (Boosting) | ~86.7% |
| Stacking Classifier | **~93.3%** |
| BERT (Transfer Learning) | ~91.1% |
| LSTM (RNN) | ~67% |

> **Note:**  
> Classical ML and ensemble models used a **20% test split**, while BERT used a **30% test split** to ensure stable validation of high-dimensional embeddings.

---

## Qualitative Analysis: Confidence & Robustness

Beyond accuracy, models were evaluated using **prediction probabilities (confidence scores)** on three inputs:
1. Clearly Human-written review  
2. Clearly AI-generated review  
3. **Humanized AI** (adversarial sample)

### Key Observations
- Most models performed well on clear Human and AI inputs.
- **TF-IDF-based models** (Random Forest, XGBoost, Stacking) were often **fooled by humanized AI**, misclassifying it as Human with **high confidence**.
- **XGBoost** showed the most concerning behavior, predicting adversarial AI as Human with ~93% confidence.
- **BERT** was the **only model** to correctly classify humanized AI as synthetic, assigning ~96% confidence.

---

## Key Takeaways
- **Highest Accuracy:** Stacking Classifier  
- **Best Real-World Robustness:** BERT  
- **Main Insight:**  
  > High validation accuracy does not guarantee robustness against adversarial or humanized AI text.

TF-IDF-based models rely heavily on surface-level lexical cues, while transformer-based models like BERT leverage deeper semantic structure.

---

## Conclusion
This project highlights a critical trade-off in synthetic text detection:

- **Statistical Efficiency:** Ensemble methods excel on standard datasets.
- **Semantic Robustness:** Transfer Learning (BERT) is essential for detecting adversarial, human-like AI text.

For real-world AI detection systems, **context-aware models are significantly more reliable** than keyword-based approaches.

---

## Files in This Repository
- `human_vs_gen_review.ipynb` – Complete experimentation notebook  
- `imdb_human_vs_gen_review.csv` – Dataset  
- `Synthetic_Text_Detection.docx` – Detailed project report  
- `README.md` – Project overview and documentation  

---

## Author
**Dhruvi Patel**  
