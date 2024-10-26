import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os , sys
import dill 

file_path = os.path.join(os.getcwd() , "netflix_dataset.csv")
df = pd.read_csv(file_path)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("model_confusion_matrix.png")
    plt.show()

def initiate_model_training(df):
    df = df.drop(["show_id", "date_added", "release_year", "title"], axis=1)

    # Encode categorical features
    label_encoders = {}
    categorical_columns = ["country", "rating", "type"]
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Handle the genres
    df['listed_in'] = df['listed_in'].fillna('').str.split(', ')
    data_exploded = df.explode('listed_in')
    genre_dummies = pd.get_dummies(data_exploded['listed_in'], prefix='genre')
    genre_encoded = genre_dummies.groupby(data_exploded.index).sum()
    df = pd.concat([df, genre_encoded], axis=1)

    # Extract duration
    df['duration'] = df['duration'].str.extract('(\d+)').astype(int)

    # Handle directors using one-hot encoding
    df['director'] = df['director'].fillna('Unknown')
    director_dummies = pd.get_dummies(df["director"], prefix="director")
    df = pd.concat([df, director_dummies], axis=1)
    df.drop(["director", "listed_in"], axis=1, inplace=True)

    # Split into features and target
    X = df.drop("type", axis=1)
    y = df["type"]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    
    param_grid={'subsample': 0.5, 'n_estimators': np.int64(150), 'max_depth': 15, 'learning_rate': 0.1, 'gamma': 0.3, 'colsample_bytree': 0.75}
    
    # Train Random Forest model
    model_xgb = XGBClassifier(random_state=42 , param_grid=param_grid)
    model_xgb.fit(X_train, y_train)

    with open("trained_model.dill", 'wb') as file:
        dill.dump(model_xgb, file)

    with open("label_encoder.dill", 'wb') as file:
        dill.dump(label_encoders, file)

    # Save feature names
    feature_names = X.columns.tolist()
    with open("feature_names.dill", 'wb') as file:
        dill.dump(feature_names, file)

    # Make predictions and evaluate the model
    rf_predict = model_xgb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, rf_predict):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_predict))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, rf_predict)


if __name__=="__main__"   :


    try:
        initiate_model_training(df)    
    except Exception as e:
        print(e) 
