# 🩺 Early Liver Disease Detection Using Machine Learning

A machine learning project for predicting the presence of **liver disease** from clinical patient data using **Logistic Regression**.

This project demonstrates a complete supervised machine learning workflow, starting from dataset loading and feature/target identification to model training and evaluation using **Accuracy, Sensitivity (Recall), and Specificity**.

> **Note:** The notebook file is currently named `Lung_disease_prediction_model.ipynb`, but the implemented project is focused on **liver disease detection** using the **Indian Liver Patient Dataset**.

---

## 📌 Project Overview

Early detection of liver disease can play an important role in improving patient outcomes and supporting clinical decision-making.

In this project, clinical data from the **Indian Liver Patient Dataset** is used to develop a binary classification model capable of predicting whether a patient has liver disease.

The project follows a straightforward machine learning pipeline:

```text
Dataset
   ↓
Data Loading
   ↓
Feature & Target Identification
   ↓
Train/Test Split
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Model Evaluation
   ├── Accuracy
   ├── Sensitivity / Recall
   └── Specificity
```

The primary goal is to demonstrate how a classical machine learning algorithm can be applied to a healthcare classification problem.

---

## 🎯 Objectives

The main objectives of this project are:

- Load and inspect the liver disease dataset.
- Understand the structure and columns of the dataset.
- Identify the target variable.
- Separate input features from the target.
- Split the dataset into training and testing subsets.
- Train a Logistic Regression classification model.
- Generate predictions on unseen test data.
- Evaluate the model using clinically relevant classification metrics.
- Understand the model's ability to identify positive and negative cases.

---

## 🧠 Machine Learning Approach

### Algorithm: Logistic Regression

The project uses **Logistic Regression** as its classification algorithm.

Logistic Regression is well suited to binary classification problems where the objective is to predict one of two possible outcomes.

In this project, the model learns a relationship between the patient's clinical features and the target outcome.

The model is initialized as:

```python
LogisticRegression(max_iter=1000)
```

The increased `max_iter` value allows the optimization process to run for up to 1,000 iterations, helping the model converge during training.

---

## 📊 Dataset

The project uses the:

**Indian Liver Patient Dataset**

The dataset is loaded from:

```text
indian_liver_patient_dataset.csv
```

The notebook loads the dataset using Pandas:

```python
import pandas as pd

diabetes_data = pd.read_csv("indian_liver_patient_dataset.csv")
```

> The variable is named `diabetes_data` in the notebook, but the dataset and project itself concern **liver disease**.

### Target Variable

The notebook defines:

```python
target_column = "Outcome"
```

Therefore:

- **Target (`y`)** → `Outcome`
- **Features (`X`)** → All columns except `Outcome`

The target values are also inspected using:

```python
diabetes_data[target_column].unique()
```

---

## 🔬 Project Workflow

### 1. Dataset Loading

The dataset is loaded using Pandas.

```python
import pandas as pd

diabetes_data = pd.read_csv("indian_liver_patient_dataset.csv")

print("Dataset loaded successfully!")
```

The notebook then checks the number of records:

```python
print("Total number of rows:", len(diabetes_data))
```

and displays the first five records:

```python
diabetes_data.head()
```

This provides an initial understanding of the dataset structure.

---

### 2. Identifying the Target

The project identifies `Outcome` as the prediction target:

```python
target_column = "Outcome"
```

The notebook also displays the available target values:

```python
print(diabetes_data[target_column].unique())
```

The target represents the classification outcome that the model is trained to predict.

---

### 3. Identifying Features

The input features are created by removing the target column:

```python
X = diabetes_data.drop(columns=[target_column])
```

The notebook then displays:

- Number of feature columns
- Names of feature columns

```python
print("Number of feature columns:", len(X.columns))
print("Feature column names:")
print(X.columns.tolist())
```

Conceptually:

```text
X = Patient Clinical Features
y = Outcome
```

---

### 4. Train/Test Split

The dataset is divided into training and testing sets using Scikit-learn's `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The configuration used is:

| Parameter | Value |
|---|---:|
| Training data | 80% |
| Testing data | 20% |
| Random state | 42 |

The `random_state=42` ensures that the split can be reproduced.

The notebook also prints the resulting dataset sizes and feature shapes.

---

## 🤖 Model Training

The selected classifier is:

```text
Logistic Regression
```

It is imported from Scikit-learn:

```python
from sklearn.linear_model import LogisticRegression
```

The model is created with:

```python
model = LogisticRegression(max_iter=1000)
```

It is then trained using:

```python
model.fit(X_train, y_train)
```

During training, the model learns patterns in the training data that can be used to predict the target outcome for previously unseen patients.

---

# 📈 Model Evaluation

After training, predictions are generated on the test set:

```python
y_pred = model.predict(X_test)
```

The project evaluates the model using three metrics:

1. Accuracy
2. Sensitivity / Recall
3. Specificity

These metrics provide a more informative view of classification performance than accuracy alone.

---

## 1. Accuracy

Accuracy measures the overall proportion of correctly classified observations.

The notebook calculates it using:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
```

Formula:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Where:

- **TP** = True Positives
- **TN** = True Negatives
- **FP** = False Positives
- **FN** = False Negatives

A higher accuracy means that the model correctly classified a larger proportion of the test samples.

---

## 2. Sensitivity / Recall

Sensitivity measures how effectively the model identifies positive cases.

The notebook calculates sensitivity using:

```python
from sklearn.metrics import recall_score

sensitivity = recall_score(y_test, y_pred)
```

Formula:

```text
Sensitivity = TP / (TP + FN)
```

In a medical prediction context, sensitivity is particularly important because false negatives can represent patients whose disease is present but not detected by the model.

---

## 3. Specificity

Specificity measures how effectively the model identifies negative cases.

The notebook first generates a confusion matrix:

```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()
```

Specificity is then calculated as:

```python
specificity = tn / (tn + fp)
```

Formula:

```text
Specificity = TN / (TN + FP)
```

A higher specificity means the model is better at correctly identifying patients who do not belong to the positive class.

---

# 🧮 Confusion Matrix

The project uses the confusion matrix as the basis for calculating specificity.

The four possible prediction outcomes are:

| | Actual Positive | Actual Negative |
|---|---:|---:|
| **Predicted Positive** | True Positive (TP) | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN) |

These values allow us to understand not only how many predictions were correct, but also **what type of mistakes the model made**.

---

# 🛠️ Technologies Used

The project is implemented in Python using the following technologies and libraries:

| Technology | Purpose |
|---|---|
| **Python** | Programming language |
| **Jupyter Notebook** | Interactive development environment |
| **Pandas** | Dataset loading and manipulation |
| **Scikit-learn** | Machine learning and evaluation |
| **Logistic Regression** | Binary classification |
| **Matplotlib / Visualization** | Not implemented in the current notebook |

### Main Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix
)
```

---

# 📁 Project Structure

A recommended repository structure is:

```text
Liver-Disease-Prediction/
│
├── 📓 Lung_disease_prediction_model.ipynb
├── 📊 indian_liver_patient_dataset.csv
├── 📄 README.md
└── 📜 LICENSE
```

> Consider renaming the notebook to `Liver_disease_prediction_model.ipynb` to match the actual project.

---

# 🚀 Getting Started

## Prerequisites

Make sure you have Python installed.

Python 3.8+ is recommended.

You can verify your Python installation with:

```bash
python --version
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR_REPOSITORY
```

---

## 2. Install Dependencies

Install the required Python packages:

```bash
pip install pandas scikit-learn jupyter
```

Alternatively:

```bash
pip install -r requirements.txt
```

A suitable `requirements.txt` would contain:

```text
pandas
scikit-learn
jupyter
```

---

## 3. Launch Jupyter Notebook

Run:

```bash
jupyter notebook
```

Then open:

```text
Lung_disease_prediction_model.ipynb
```

---

## 4. Add the Dataset

Make sure the dataset is located in the same directory as the notebook:

```text
indian_liver_patient_dataset.csv
```

The notebook expects this exact filename:

```python
pd.read_csv("indian_liver_patient_dataset.csv")
```

---

# ▶️ Running the Project

The notebook is organized into four major tasks:

### Task 1 — Analyze the Dataset

- Load the dataset
- Check whether it loaded successfully
- Count the number of rows
- Display the first five records

### Task 2 — Identify Features and Target

- Display dataset columns
- Identify `Outcome` as the target
- Inspect target values
- Create feature matrix `X`
- Create target vector `y`
- Split the dataset into training and testing sets

### Task 3 — Train the Classifier

- Select Logistic Regression
- Create the model
- Train the model on the training dataset

### Task 4 — Evaluate the Model

- Generate predictions
- Calculate accuracy
- Calculate sensitivity / recall
- Calculate specificity

---

# 📋 Results

The notebook calculates the following evaluation metrics:

```text
Accuracy
Sensitivity (Recall)
Specificity
```

The exact numerical results should be obtained by executing the notebook against the dataset.

### Results Table

| Metric | Result |
|---|---:|
| Accuracy | Run notebook |
| Sensitivity / Recall | Run notebook |
| Specificity | Run notebook |

This repository intentionally does not hard-code performance values that are not saved in the notebook.

---

# 🩻 Why These Metrics Matter

For healthcare-related machine learning, relying solely on accuracy can be misleading.

A model could achieve reasonable accuracy while still missing a significant number of positive cases.

That's why this project evaluates both:

### Sensitivity

> "Of the patients who actually belong to the positive class, how many did the model identify?"

### Specificity

> "Of the patients who actually belong to the negative class, how many did the model correctly identify?"

Considering both provides a more meaningful view of classification behavior.

---

# ⚠️ Important Medical Disclaimer

This project is intended for **educational and research purposes only**.

It is a demonstration of a machine learning classification workflow and **is not a clinically validated diagnostic system**.

The predictions produced by this model should not be used to:

- Diagnose a patient
- Replace a physician
- Make medical treatment decisions
- Determine whether someone has liver disease
- Serve as a substitute for clinical testing

Real-world medical AI systems require extensive validation, appropriate datasets, clinical expertise, regulatory review, bias assessment, and testing on independent populations before they can be considered for clinical use.

---

# 🔍 Current Limitations

The current implementation is intentionally simple and serves primarily as a demonstration of a machine learning workflow.

Some potential limitations include:

### 1. No documented preprocessing pipeline

The current notebook does not explicitly implement a dedicated preprocessing pipeline for handling issues such as missing values, categorical encoding, or feature scaling.

### 2. Single model

Only Logistic Regression is evaluated.

Other algorithms could potentially be explored and compared.

### 3. Single train/test split

The project uses an 80/20 split with `random_state=42`.

Cross-validation could provide a more robust estimate of model performance.

### 4. Limited evaluation metrics

The current notebook evaluates:

- Accuracy
- Sensitivity
- Specificity

Additional metrics could provide a broader understanding of model performance.

### 5. No hyperparameter optimization

The Logistic Regression model uses:

```python
max_iter=1000
```

but no systematic hyperparameter search is implemented.

### 6. No external validation

The current workflow does not demonstrate evaluation on an independent external dataset.

For medical applications, external validation is especially important.

---

# 🚀 Future Improvements

This project can be expanded significantly.

Possible improvements include:

- [ ] Add comprehensive exploratory data analysis (EDA)
- [ ] Handle missing values explicitly
- [ ] Encode categorical variables where necessary
- [ ] Apply feature scaling where appropriate
- [ ] Add cross-validation
- [ ] Compare multiple classification algorithms
- [ ] Perform hyperparameter tuning
- [ ] Generate a visual confusion matrix
- [ ] Add ROC-AUC evaluation
- [ ] Add Precision and F1-score
- [ ] Analyze feature importance
- [ ] Evaluate class imbalance
- [ ] Build a reusable preprocessing pipeline
- [ ] Test the model on an independent dataset
- [ ] Create a simple prediction interface
- [ ] Deploy the trained model as an API
- [ ] Build a web-based demonstration interface

---

# 🧪 Potential Model Comparison

A future version could compare Logistic Regression with models such as:

```text
Logistic Regression
        │
        ├── Decision Tree
        ├── Random Forest
        ├── Support Vector Machine
        ├── K-Nearest Neighbors
        └── Gradient Boosting
```

The models could then be compared using:

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| SVM | — | — | — | — | — |

This would make it easier to determine which model provides the most useful performance for the dataset.

---

# 💡 Learning Outcomes

This project demonstrates several important machine learning concepts:

- Binary classification
- Feature and target separation
- Train/test splitting
- Logistic Regression
- Model training
- Prediction generation
- Accuracy calculation
- Recall / sensitivity
- Confusion matrices
- Specificity
- Basic healthcare machine learning workflows

It provides a foundation for developing more advanced medical machine learning systems.

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature/improvement
```

3. Make your changes.
4. Commit your changes.

```bash
git commit -m "Add model evaluation improvements"
```

5. Push the branch.

```bash
git push origin feature/improvement
```

6. Open a Pull Request.

---

# 📜 License

If this project is intended to be open source, add an appropriate license to the repository.

For example:

```text
MIT License
```

The license should be added as a separate `LICENSE` file in the repository.

---

# 👨‍💻 Author

**Mughees**

Student • Full-Stack Developer • Machine Learning Enthusiast

Interested in:

- Artificial Intelligence
- Machine Learning
- Medical AI
- Healthcare Technology
- Full-Stack Development
- Research & Innovation

---

# ⭐ Project Summary

**Early Liver Disease Detection Using Machine Learning** demonstrates how clinical patient data can be used to build a binary classification model using Logistic Regression.

The project covers the fundamental machine learning pipeline:

```text
Clinical Data
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Logistic Regression
     ↓
Predictions
     ↓
Accuracy
Sensitivity
Specificity
```

Although this implementation is a relatively simple baseline, it provides a solid starting point for experimenting with **machine learning in healthcare** and can be extended into a more comprehensive medical AI project through better preprocessing, model comparison, validation, interpretability, and deployment.