# Telco Customer Churn Predictor 🔮

An interactive machine learning application built with **Streamlit** and **XGBooThis project aims to predict customer churn in a telecommunications company using machine learning techniques. It includes a complete pipeline from data cleaning and exploratory data analysis (EDA) to model training and deployment via an interactive Streamlit application.

## Business Context
Customer churn (attrition) is a critical metric for subscription-based businesses like telecommunications companies. It is significantly more expensive to acquire a new customer than to retain an existing one. By identifying customers who are at high risk of churning, the company can:
*   **Proactively intervene** with targeted retention campaigns (e.g., discounts, special offers).
*   **Improve customer satisfaction** by addressing pain points.
*   **Optimize revenue** by reducing the loss of recurring income.

This project enables the company to move from reactive to proactive churn management.

## Key Performance Metrics
To evaluate the models, we focus on the following metrics:
*   **Accuracy:** The ratio of correctly predicted observations to the total observations. It gives a general idea of model performance but can be misleading in imbalanced datasets.
*   **F1-Score:** The weighted average of Precision and Recall. It is useful when the class distribution is uneven (churners vs. non-churners), as it balances false positives and false negatives.
*   **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the model's ability to distinguish between classes. A higher AUC (closer to 1) allows the model to better rank customers by their churn probability.

## Model Comparison
We evaluated three different models to identify the best performer for this task:

| Model | Accuracy | F1-Score (Weighted) | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 74% | 0.75 | **0.84** |
| **Random Forest** | **78%** | **0.77** | 0.81 |
| **XGBoost** | 76% | **0.77** | 0.82 |

### Analysis
*   **Logistic Regression** achieved the highest **ROC-AUC (0.84)**, suggesting it is excellent at ranking customers by risk, even if its hard classification accuracy is slightly lower. It provides good interpretability.
*   **Random Forest** yielded the highest **Accuracy (78%)**, making it a strong candidate for overall correctness.
*   **XGBoost** offered a balanced performance with high F1-score and valid ROC-AUC. It handles complex non-linear relationships well and is often robust against overfitting.

*Note: The deployed application primarily uses the **XGBoost** model due to its speed, performance balance, and ability to handle various data features effectively.*od of churn in real-time.

## Project Overview

Customer churn is a critical metric for businesses. This project leverages historical customer data to train a predictive model that identifies customers at high risk of leaving. The interactive dashboard provides actionable insights by allowing stakeholders to simulate different customer profiles and see the impact on churn probability.

## Features

- **Interactive Dashboard**: Built with Streamlit for easy user interaction.
- **Real-Time Predictions**: Instant churn probability estimates based on user inputs.
- **Advanced Modeling**: Utilizes an XGBoost classifier optimized for imbalanced data.
- **Comprehensive Feature Engineering**: Includes tenure binning, charges per tenure calculation, and family status derivation.
- **Modular Codebase**: Clean separation of data processing, training, and application logic.

## Tech Stack

- **Python 3.10+**
- **Streamlit**: Web application framework.
- **XGBoost**: Gradient boosting library for classification.
- **Scikit-learn**: Data preprocessing and model evaluation.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Plotly**: Interactive visualizations.

## Project Structure

```
TelcoChurnPredictor/
├── Data/                   # Dataset files (cleanedData.csv)
├── models/                 # Saved trained models (xgb_model.joblib)
├── notebooks/              # Jupyter notebooks for EDA and experimentation
│   ├── DataCleaning&EDA_TelcoData.ipynb
│   └── Prediction_TelcoData.ipynb
├── src/                    # Source code for the application
│   ├── __init__.py
│   ├── app.py              # Streamlit application entry point
│   ├── data_processing.py  # Data loading and preprocessing functions
│   └── train.py            # Model training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup & Installation

1.  **Clone the Repository** (if applicable) or navigate to the project directory:
    ```bash
    cd TelcoChurnPredictor
    ```

2.  **Create a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Running the Interactive App

To launch the Streamlit dashboard:

```bash
streamlit run src/app.py
```

The application will open in your default web browser at `http://localhost:8501`. Use the sidebar to adjust customer parameters and click **"Predict Churn"** to see the results.

### 2. Training the Model

If you need to retrain the model or regenerate the `xgb_model.joblib` file:

```bash
python src/train.py
```

This script will:
1.  Load data from `Data/cleanedData.csv`.
2.  Preprocess features (encodings, scaling, new feature creation).
3.  Train the XGBoost classifier.
4.  Evaluate performance on a test set.
5.  Save the trained pipeline to `models/xgb_model.joblib`.

## Data Description

The model uses the Telco Customer Churn dataset, which includes:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents.
- **Services**: Phone Service, Multiple Lines, Internet Service, Online Security, etc.
- **Account Info**: Contract, Paperless Billing, Payment Method.
- **Charges**: Monthly Charges, Total Charges, Tenure.

## Future Improvements

- **Deploy to Cloud**: Host the app on Streamlit Cloud or Heroku for public access.
- **Model Explainability**: Integrate SHAP values to explain *why* a specific customer is predicted to churn.
- **Batch Prediction**: Add functionality to upload a CSV file and get predictions for multiple customers at once.
- **Model Monitoring**: Track prediction drift over time.
