import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys ,os

def initiate_model_training():
    

    # Preprocess the dataset
    df = df.drop(["show_id", "date_added", "release_year", "title"], axis=1)

    # Encode categorical features
    label_encoders = {}
    categorical_columns = ["country", "rating", "type"]
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Save the encoder

    # Handle the genres
    df['listed_in'] = df['listed_in'].fillna('')
    data_exploded = df.explode('listed_in')
    genre_dummies = pd.get_dummies(data_exploded['listed_in'], prefix='genre')
    genre_encoded = genre_dummies.groupby(data_exploded.index).sum()
    df = pd.concat([df, genre_encoded], axis=1)

    # Extract duration
    df['duration'] = df['duration'].str.extract('(\d+)').astype(int)

    # Handle directors using one-hot encoding
    df['director'] = df['director'].fillna('Unknown')
    director_dummies = pd.get_dummies(df["director"], prefix="director")
    data = pd.concat([df, director_dummies], axis=1)
    data.drop(["director", "listed_in"], axis=1, inplace=True)

    # Split into features and target
    X = data.drop("type", axis=1)
    y = data["type"]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

   # Train Random Forest model
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)

  # Save the trained model, scaler, encoders, and feature names
    with open("trained_model.pkl", 'wb') as file:
       pickle.dump(model_rf, file)

    with open("scaler.pkl", 'wb') as file:
        pickle.dump(scaler, file)

    with open("label_encoder.pkl", 'wb') as file:
        pickle.dump(label_encoders, file)

    # Save feature names
    feature_names = X.columns.tolist()
    with open("feature_names.pkl", 'wb') as file:
        pickle.dump(feature_names, file)

    # Make predictions and evaluate the model
    rf_predict = model_rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, rf_predict):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_predict))

    # Plot confusion matrix
    def plot_confusion_matrix(y_true, y_pred):
       cm = confusion_matrix(y_true, y_pred)
       plt.figure(figsize=(10, 7))
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
       plt.xlabel('Predicted')
       plt.ylabel('Actual')
       plt.title('Confusion Matrix')
       plt.show()

    plot_confusion_matrix(y_test, rf_predict)

try:
    initiate_model_training()

except Exception as e:
    print(e,sys)   