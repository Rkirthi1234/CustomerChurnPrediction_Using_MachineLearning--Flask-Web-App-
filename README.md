# CustomerChurnPrediction_Using_MachineLearning--Flask-Web-App-

# Project Description:
This project is a machine learning-based web application that predicts whether a customer will churn (leave a service) or stay, based on their attributes and service usage patterns. Built using Python, Scikit-learn, and Flask, this project demonstrates how predictive modeling can be integrated into a user-friendly web interface.

The model is trained using the Telco Customer Churn dataset, and leverages a Random Forest Classifier with SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance. It achieves a high accuracy score on the test set, making it a reliable solution for churn prediction in customer service industries such as telecom, banking, or subscription services.

# Features:
Machine Learning Model trained with oversampled data for better minority class prediction

Prediction of customer churn based on 19 key features

Simple and elegant Flask web interface

Clean UI styled with CSS

Displays clear prediction results with supportive insights

# Technologies Used:
Python 3.x

Pandas, NumPy, Scikit-learn, Imbalanced-learn

Flask (Backend web framework)

HTML5 & CSS3 (Frontend)

Pickle (Model serialization)

# Machine Learning Details:
Model Used: Random Forest Classifier

Data Handling: One-Hot Encoding, Feature Engineering (e.g., Tenure binning)

Imbalanced Data Solution: SMOTE for synthetic data generation

Model Evaluation: Accuracy Score, Precision, Recall, F1-score

# How it Works:
User enters customer-related details into the web form.

Flask app takes the inputs, processes them into a feature array.

The pre-trained model (loaded via Pickle) predicts whether the customer is likely to churn.

The result is displayed on the same web page with a message like:

"This customer is likely to churn."

"This customer is not likely to churn."

# Result:
<img width="1577" height="909" alt="image" src="https://github.com/user-attachments/assets/7896be39-b6a6-4a9c-8789-e7eebfc75aeb" />

<img width="1577" height="969" alt="image" src="https://github.com/user-attachments/assets/3a984ccc-c304-40c1-9b90-1bd326fc9476" />

