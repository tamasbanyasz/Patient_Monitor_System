
"""
Imports libraries:

    -   joblib to load saved models.

    -   pandas for data manipulation (loading CSV file).

    -   matplotlib.pyplot for plotting results.

    -   Various modules from sklearn for evaluation metrics, train-test splitting, and label encoding.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
Loads the dataset:

    -   Reads the CSV file szintetikus_betegadatok.csv into a pandas DataFrame.

    -   Splits the data into features (X) and target variable (y), where Gyógyszer is the target.
"""
df = pd.read_csv('szintetikus_betegadatok.csv')  

X = df.drop('Gyógyszer', axis=1)
y = df['Gyógyszer']

"""
Label encoding:

    -   Encodes the categorical target labels into numeric codes for internal use but keeps the original labels for evaluation.
    
"""

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

"""
Train-test split:

    -   Splits the dataset into training and testing sets with an 80-20 ratio, keeping the original label distribution (stratified split).

    -   Note: The target labels y remain in their original text form for clarity.

"""

# Train-test split - IMPORTANT: y should be the original labels (text), not encoded!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

"""
Loads saved models:

    -   Loads three pre-trained classifiers from disk: Random Forest, SVM, and XGBoost.
"""

# Model file paths
model_files = {
    "Random Forest": "Random_Forest_model.pkl",
    "SVM": "SVM_model.pkl",
    "XGBoost": "XGBoost_model.pkl"
}

results = {}

for model_name, file_path in model_files.items():
    model = joblib.load(file_path)
    y_pred = model.predict(X_test)
    
    """
    Makes predictions and evaluates:

        -   For each model, it predicts the labels on the test set.

        -   Checks the prediction type—if numeric codes, it converts them back to the original string labels.

        -   Calculates the accuracy score and prints a detailed classification report including precision, recall, and F1-score.
    """

    # Debug info about types and values
    print(f"\n{model_name} prediction type: {type(y_pred[0])}, first 5 predictions: {y_pred[:5]}")
    print(f"True label type: {type(y_test.iloc[0])}, first 5 true labels: {y_test.iloc[:5].to_list()}")

    # If predictions are numeric (encoded), convert back to string labels
    if not isinstance(y_pred[0], str):
        y_pred_labels = label_encoder.inverse_transform(y_pred)
    else:
        y_pred_labels = y_pred

    # y_test is still a pandas Series here, convert to list of strings
    y_test_labels = y_test.to_list()

    acc = accuracy_score(y_test_labels, y_pred_labels)
    results[model_name] = acc * 100

    print(f"\n Model: {model_name}")
    print(f" Accuracy: {acc:.2%}")
    print(" Report:\n", classification_report(y_test_labels, y_pred_labels))

"""
Visualizes the results:

    -   Plots a bar chart comparing the accuracy percentages of the three models.

    -   Adds the exact accuracy value above each bar for readability.

"""
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['green', 'blue', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
for i, v in enumerate(results.values()):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
