import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

"""
The codes for the "compare_ml_models.py" and "compare_previous_ml_results.py" files were written  in Jupiter Notebook at first.

"""

# Load data
df = pd.read_csv("szintetikus_betegadatok.csv")

# Outlier removal based on IQR (only for numeric columns)
def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

# Continue working only with data cleaned from outliers
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df_clean = remove_outliers_iqr(df, numerical_cols)

df_clean = df_clean.reset_index(drop=True)

print(f"Number of records after outlier removal: {len(df_clean)} / {len(df)}")


# New X and y values
X = df_clean.drop(columns=['Gyógyszer'])
y = df_clean['Gyógyszer']


# Label encoding only for XGBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Same, but with encoded labels for XGBoost:
y_train_enc = y_encoded[y_train.index.to_numpy()]
y_test_enc = y_encoded[y_test.index.to_numpy()]

# Column transformation
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Models and hyperparameters
models = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [None, 10],
        }
    ),
    "SVM": (
        SVC(probability=True),
        {
            'classifier__C': [1.0, 10.0],
            'classifier__kernel': ['rbf']
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric='mlogloss'),
        {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.1]
        }
    )
}

# Model comparison
best_models = {}

for name, (clf, param_grid) in models.items():
    print(f"\n Model: {name}")

    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )

    # Fit separately for XGBoost because it needs encoded labels
    if name == "XGBoost":
        grid_search.fit(X_train, y_train_enc)
        y_pred = grid_search.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_eval = y_test.values
    else:
        grid_search.fit(X_train, y_train)
        y_pred_labels = grid_search.predict(X_test)
        y_test_eval = y_test
        
    acc = accuracy_score(y_test_eval, y_pred_labels)
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Best parameters:", grid_search.best_params_)
    print("Detailed report:\n", classification_report(y_test_eval, y_pred_labels))

    # Save the model
    best_models[name] = grid_search.best_estimator_
    joblib.dump(grid_search.best_estimator_, f"{name.replace(' ', '_')}_model.pkl")
    print(f"Saved: {name.replace(' ', '_')}_model.pkl")

