# Cross-Project Software Defect Prediction Using Machine Learning Techniques

Machine Learning-based software defect prediction using public PROMISE repository datasets.

## Project Overview

This project investigates cross-project software defect prediction using machine learning techniques on public software quality datasets from the PROMISE repository.

The objective is to predict whether a software module is defective based on software metrics collected from different open-source software projects.

Five projects were selected and combined into a unified dataset:

* Ant
* Camel
* JEdit
* POI
* Xalan

Several machine learning models, ensemble methods, and optimization strategies were trained and compared to identify the most effective approach for cross-project defect prediction.

---

## Dataset

### Source

PROMISE Software Engineering Repository

### Projects Included

| Project |
| ------- |
| Ant     |
| Camel   |
| JEdit   |
| POI     |
| Xalan   |

### Target Variable

The original bug count was transformed into a binary classification problem:

```python
bug > 0  -> Defective (1)
bug = 0  -> Non-defective (0)
```

---

## Models Evaluated

The following machine learning approaches were implemented and compared:

1. Random Forest
2. Random Forest + SMOTE
3. XGBoost + SMOTE
4. Random Forest + XGBoost Ensemble
5. Weighted Ensemble
6. Tuned Random Forest + SMOTE

---

## Methodology

### Data Preparation

* Combined multiple project datasets into a single dataset
* Added project and version information
* Converted bug counts into binary labels
* Removed non-feature columns before training
* Applied an 85%-15% train-test split for evaluation

### Handling Class Imbalance

The dataset contained fewer defective modules than non-defective modules. To address this issue, SMOTE (Synthetic Minority Oversampling Technique) was applied to the training data.

### Hyperparameter Optimization

Random Forest hyperparameters were optimized through systematic experimentation using different combinations of:

* Number of trees
* Maximum tree depth
* Minimum samples per leaf
* Class weighting

### Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

## Results

| Model                       | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --------------------------- | -------: | --------: | -----: | -------: | ------: |
| Random Forest               |    0.751 |     0.403 |  0.420 |    0.411 |   0.736 |
| Random Forest + SMOTE       |    0.708 |     0.377 |  0.631 |    0.472 |   0.728 |
| XGBoost + SMOTE             |    0.702 |     0.369 |  0.623 |    0.464 |   0.716 |
| RF + XGBoost Ensemble       |    0.741 |     0.412 |  0.589 |    0.485 |   0.733 |
| Weighted Ensemble           |    0.744 |     0.411 |  0.549 |    0.470 |   0.737 |
| Tuned Random Forest + SMOTE |    0.736 |     0.416 |  0.677 |    0.515 |   0.756 |

### Best Performing Model

The optimized Random Forest model trained on SMOTE-balanced data achieved the best overall performance.

* F1-score = 0.515
* Recall = 0.677
* ROC-AUC = 0.756

These results demonstrate that balancing the dataset and tuning model parameters significantly improved defect prediction performance.

---

## Feature Importance

The feature importance analysis of the optimized Random Forest model identified the following software metrics as the most influential predictors of software defects:

1. LOC
2. AMC
3. DAM
4. LCOM3
5. MFA
6. RFC
7. NPM
8. CBM
9. AVG_CC
10. CAM

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Jupyter Notebook

---

## Repository Structure

```text
datasets/
├── ant/
├── camel/
├── jedit/
├── poi/
└── xalan/

notebooks/
└── 01_cross_project_defect_prediction.ipynb

results/
paper/
```

---

## Future Improvements

Potential future work includes:

* Testing additional ensemble strategies
* Incorporating LightGBM and CatBoost models
* Performing cross-validation across project combinations
* Applying automated hyperparameter optimization techniques
* Exploring deep learning approaches for defect prediction
* Evaluating additional PROMISE repository projects

---

## Author

Nazlıcan Güner

M.Eng. Computer Science (Software Engineering)

GISMA University of Applied Sciences

Berlin, Germany
