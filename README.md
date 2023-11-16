# 88612025_Churning_Customers
 Churning Customers in a Telecoms Company
 
# Neural Networks for Churn Prediction

## Summary

The goal of this research is to anticipate customer turnover using a neural network-based machine learning model. The dataset utilized comprises information on numerous customer characteristics, and the purpose is to create a model that reliably predicts whether a client would churn.

## Directory and File Structure

- Jupyter Notebook contains the primary code for data preparation, exploratory data analysis (EDA), feature engineering, and model development ('Churn_Prediction.ipynb').
- 'CustomerChurn_dataset.csv': a CSV file containing the raw dataset with customer data.
- 'churn_model.pkl': A Joblib file that contains the trained and tweaked neural network model.
-'README.md': This is the readme file.

## Code Synopsis

1. **Data Loading and Exploration:** The dataset is loaded from Google Drive, and some basic exploration is carried out.
   
2. **Data Preprocessing:** Identifies and labels numerical and category characteristics.
   - To translate category information into numerical representations, label encoding is used.
   - To comprehend the correlations between characteristics, a correlation matrix is computed.
   - The top attributes that are connected with the target variable ('Churn') are discovered.

3. **Feature Scaling:** To normalize the values of the specified characteristics, standard scaling is done.

4. **Data Splitting:** The dataset is divided into two parts: training and testing.

5. **Neural Network Model:** Keras is used to create a basic Multi-Layer Perceptron (MLP) neural network.
   To optimize hyperparameters, grid search with cross-validation is employed.

6. **Model Evaluation:** On the test set, the best model is evaluated, and accuracy and ROC AUC values are computed.

7. **Retraining:** Using the best hyperparameters, the best model is retrained on the full dataset.

8. **Model Saving:** For future usage, the trained model is stored as a joblib file.

## Usage

1. In a Jupyter notebook, open and execute the 'Churn_Prediction.ipynb' notebook.
2. Make sure the necessary libraries are installed ('pandas', 'numpy','scikit-learn','seaborn','matplotlib', 'keras','scikeras').
3. If required, change the paths and filenames.
4. Follow the code comments and run the cells in the correct order.

## Prerequisites

Jupyter Notebook - Python 3.x
- Libraries:'pandas','numpy','scikit-learn','seaborn','matplotlib','keras','scikeras'

## Writer

[Your Surname]

## Appreciation

- This project is based on [source link]'s real-world dataset.
- Thank you so much to [mentor, coworker, etc.] for your guidance and assistance.

Feel free to add additional parts or information based on the specifics of your project.




