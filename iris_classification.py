# ============================================================
# Iris Classification Model Comparison
# Author: Danielle Walker
# ============================================================

# This program loads the Iris dataset and compares six
# classification approaches:
#
# 1. Decision Tree
# 2. Rule-Based Classifier
# 3. Gaussian Naive Bayes
# 4. Logistic Regression
# 5. K-Nearest Neighbors
# 6. Support Vector Machine
#
# Each model is evaluated using:
# Accuracy, Confusion Matrix, Precision, Recall, and F1-score.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


print("Libraries imported successfully.")


# ------------------------------------------------------------
# 2. LOAD THE IRIS DATASET
# ------------------------------------------------------------

iris = load_iris()

# X contains the flower measurements
X = iris.data

# y contains the flower species labels
y = iris.target


# ------------------------------------------------------------
# 3. DISPLAY DATASET CHARACTERISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("IRIS DATASET CHARACTERISTICS")
print("=" * 60)

print("\nFeature names:")
print(iris.feature_names)

print("\nTarget classes:")
print(iris.target_names)

print("\nDataset shape:")
print(X.shape)

print("\nNumber of samples:")
print(X.shape[0])

print("\nNumber of features:")
print(X.shape[1])


# Display the number of observations in each class
unique_classes, class_counts = np.unique(y, return_counts=True)

print("\nClass distribution:")

for class_number, count in zip(unique_classes, class_counts):
    print(f"{iris.target_names[class_number]}: {count}")


# ------------------------------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

# 80% of the data is used for training.
# 20% of the data is used for testing.
#
# random_state=42 makes the results reproducible.
# stratify=y maintains equal class representation.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAINING AND TESTING DATA")
print("=" * 60)

print("\nTraining feature set:", X_train.shape)
print("Testing feature set:", X_test.shape)

print("Training target set:", y_train.shape)
print("Testing target set:", y_test.shape)


# Display test class distribution
unique_test_classes, test_counts = np.unique(
    y_test,
    return_counts=True
)

print("\nTest class distribution:")

for class_number, count in zip(
    unique_test_classes,
    test_counts
):
    print(f"{iris.target_names[class_number]}: {count}")


# ------------------------------------------------------------
# 5. CREATE A LIST TO STORE MODEL RESULTS
# ------------------------------------------------------------

results = []


# ------------------------------------------------------------
# 6. CREATE MODEL EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Trains a classification model, makes predictions,
    calculates evaluation metrics, prints the results,
    and stores the results for later comparison.
    """

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Calculate weighted precision
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Calculate weighted recall
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Calculate weighted F1-score
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Print model results
    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names,
            digits=3,
            zero_division=0
        )
    )

    print(
        "Accuracy:",
        round(accuracy, 3)
    )

    print(
        "Weighted Precision:",
        round(precision, 3)
    )

    print(
        "Weighted Recall:",
        round(recall, 3)
    )

    print(
        "Weighted F1-score:",
        round(f1, 3)
    )

    # Save the results for the comparison table
    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
    )

    return y_pred


# ============================================================
# PART 2A
# RULE-BASED AND PROBABILISTIC MODELS
# ============================================================


# ------------------------------------------------------------
# 7. DECISION TREE CLASSIFIER
# ------------------------------------------------------------

decision_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

decision_tree_predictions = evaluate_model(
    "Decision Tree",
    decision_tree,
    X_train,
    X_test,
    y_train,
    y_test
)


# ------------------------------------------------------------
# 8. CUSTOM RULE-BASED CLASSIFIER
# ------------------------------------------------------------

class SimpleRuleBasedClassifier:
    """
    A simple classifier that predicts Iris species
    using manually defined if-else rules.
    """

    def fit(self, X, y=None):
        # This classifier uses predefined rules,
        # so no mathematical training is required.
        return self

    def predict(self, X):

        predictions = []

        for flower in X:

            sepal_length = flower[0]
            sepal_width = flower[1]
            petal_length = flower[2]
            petal_width = flower[3]

            # Rule 1:
            # Iris Setosa generally has very short petals.
            if petal_length < 2.5:
                predictions.append(0)

            # Rule 2:
            # Versicolor generally has narrower petals
            # than Virginica.
            elif petal_width < 1.75:
                predictions.append(1)

            # Rule 3:
            # Otherwise classify the flower as Virginica.
            else:
                predictions.append(2)

        return np.array(predictions)


rule_based_model = SimpleRuleBasedClassifier()

rule_based_predictions = evaluate_model(
    "Rule-Based Classifier",
    rule_based_model,
    X_train,
    X_test,
    y_train,
    y_test
)


# ------------------------------------------------------------
# 9. GAUSSIAN NAIVE BAYES
# ------------------------------------------------------------

naive_bayes = GaussianNB()

naive_bayes_predictions = evaluate_model(
    "Gaussian Naive Bayes",
    naive_bayes,
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# PART 2B
# DISTANCE AND OPTIMIZATION-BASED MODELS
# ============================================================


# ------------------------------------------------------------
# 10. LOGISTIC REGRESSION
# ------------------------------------------------------------

# StandardScaler is included in the pipeline so that
# features are standardized before Logistic Regression.

logistic_regression = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_predictions = evaluate_model(
    "Logistic Regression",
    logistic_regression,
    X_train,
    X_test,
    y_train,
    y_test
)


# ------------------------------------------------------------
# 11. K-NEAREST NEIGHBORS
# ------------------------------------------------------------

# KNN depends on distances between observations,
# so feature scaling is important.

knn_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]
)

knn_predictions = evaluate_model(
    "K-Nearest Neighbors",
    knn_model,
    X_train,
    X_test,
    y_train,
    y_test
)


# ------------------------------------------------------------
# 12. SUPPORT VECTOR MACHINE
# ------------------------------------------------------------

# SVM uses an RBF kernel.
# StandardScaler standardizes the features first.

svm_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale"
            )
        )
    ]
)

svm_predictions = evaluate_model(
    "Support Vector Machine",
    svm_model,
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# PART 3
# FINAL MODEL COMPARISON
# ============================================================


# ------------------------------------------------------------
# 13. CREATE MODEL COMPARISON TABLE
# ------------------------------------------------------------

comparison_table = pd.DataFrame(results)


# Convert decimal metric values to percentages
percentage_table = comparison_table.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]:
    percentage_table[column] = (
        percentage_table[column] * 100
    ).round(2)


print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    percentage_table.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 14. IDENTIFY HIGHEST ACCURACY
# ------------------------------------------------------------

highest_accuracy = comparison_table[
    "Accuracy"
].max()

best_models = comparison_table[
    comparison_table["Accuracy"] == highest_accuracy
]["Model"].tolist()


print("\nHighest Accuracy:")

print(
    f"{highest_accuracy * 100:.2f}%"
)

print("\nModels with the highest accuracy:")

for model in best_models:
    print("-", model)


# ------------------------------------------------------------
# 15. DISPLAY FINAL ANALYSIS MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    """
The Iris dataset contains three balanced flower classes
and four numerical features.

Setosa is relatively easy to distinguish from the other
two species. Most classification errors occur between
Versicolor and Virginica because their physical
measurements overlap more closely.

The model comparison demonstrates that several different
classification techniques can perform well on a small,
balanced, low-dimensional dataset.
"""
)


print("Program completed successfully.")
