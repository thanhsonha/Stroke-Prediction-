# Stroke Prediction Analysis

## Project Overview

This project analyzes patient health and lifestyle factors to understand patterns associated with stroke risk and to build machine learning models that predict whether a patient is likely to experience a stroke.

The project uses the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset), which contains patient-level information such as age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status, and stroke outcome.

## Objective

The main goals of this project are to:

- Explore relationships between patient health factors and stroke occurrence
- Visualize stroke distribution across demographic and medical variables
- Clean and preprocess the dataset for machine learning
- Handle class imbalance using SMOTE oversampling
- Train and compare multiple classification models
- Identify the most important features related to stroke prediction

## Dataset

The dataset contains **5,110 patient records** with the following variables:

| Column | Description |
|---|---|
| `id` | Unique patient identifier |
| `gender` | Patient gender |
| `age` | Patient age |
| `hypertension` | Whether the patient has hypertension |
| `heart_disease` | Whether the patient has heart disease |
| `ever_married` | Marital status |
| `work_type` | Type of employment |
| `Residence_type` | Urban or rural residence |
| `avg_glucose_level` | Average glucose level in blood |
| `bmi` | Body mass index |
| `smoking_status` | Smoking history |
| `stroke` | Target variable: 1 = stroke, 0 = no stroke |

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn / SMOTE
- Jupyter Notebook

## Project Workflow

### 1. Data Exploration

The dataset was inspected for:

- Data types
- Missing values
- Summary statistics
- Distribution of numerical and categorical variables

### 2. Exploratory Data Analysis

Visualizations were created to analyze stroke distribution by:

- Gender
- Marital status
- Residence type
- Work type
- Smoking status
- BMI category
- Age group
- Average glucose level
- Hypertension
- Heart disease

The dataset is highly imbalanced, with approximately **4.87% stroke cases** and **95.13% non-stroke cases**.

### 3. Data Preprocessing

Preprocessing steps included:

- Filling missing BMI values
- Encoding categorical variables
- Creating dummy variables
- Removing unnecessary variables
- Scaling numerical variables
- Applying SMOTE to address class imbalance
- Splitting the data into training and testing sets

### 4. Modeling

Several machine learning models were trained and evaluated:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 84.47% |
| Linear Regression | 84.32% |
| Naive Bayes | 58.41% |
| Decision Tree | 88.69% |
| Support Vector Classifier | 89.46% |
| Random Forest | 95.58% |
| K-Nearest Neighbors | 90.49% |
| Gradient Boosting | 86.79% |

## Results

The **Random Forest model** achieved the highest accuracy at approximately **95.58%**.

Feature importance analysis showed that the most influential predictors were:

1. Age
2. BMI
3. Average glucose level

These features contributed the most to the model's stroke prediction performance.

## Key Insights

- The dataset is strongly imbalanced, making accuracy alone potentially misleading.
- Age was the most important feature in predicting stroke risk.
- BMI and average glucose level were also important predictors.
- Random Forest performed best among the tested models.
- Handling class imbalance with SMOTE helped improve model training on minority stroke cases.

## Limitations

- The dataset is highly imbalanced, with stroke cases representing less than 5% of the data.
- Accuracy may not fully reflect model performance for the minority stroke class.
- Additional evaluation metrics such as recall, precision, F1-score, and ROC-AUC should be emphasized in future improvements.
- The model should not be used for medical diagnosis without clinical validation.

## Future Improvements

Potential next steps include:

- Evaluate models using recall, precision, F1-score, and ROC-AUC
- Tune hyperparameters for Random Forest and Gradient Boosting
- Compare SMOTE with other imbalance-handling techniques
- Build a simple prediction dashboard or web app
- Add model explainability using SHAP or permutation importance
- Package the workflow into reusable scripts

## Repository Structure

```text
.
├── Stroke Project (5).ipynb
├── healthcare-dataset-stroke-data.csv
└── README.md
```

## How to Run

1. Clone this repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

3. Open the notebook:

```bash
jupyter notebook
```

4. Run `Stroke Project (5).ipynb`.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for healthcare analytics, including data exploration, visualization, preprocessing, class imbalance handling, model comparison, and feature importance analysis. The Random Forest model achieved the best performance, with age, BMI, and average glucose level emerging as the strongest predictors of stroke risk.
