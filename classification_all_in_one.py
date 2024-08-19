from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def algo_test(X, y):
    models = [
        BernoulliNB(),
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
        AdaBoostClassifier(),
        MultinomialNB()
    ]
    model_names = [
        "BernoulliNB",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "KNeighborsClassifier",
        "AdaBoostClassifier",
        "MultinomialNB"
    ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    fitted_models = []

    print("Training models...")
    for model, model_name in zip(models, model_names):
        print(f"{model_name} model is being trained...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted_models.append(model)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average="micro"))
        recall.append(recall_score(y_test, y_pred, average="micro"))
        f1.append(f1_score(y_test, y_pred, average="micro"))
        print(confusion_matrix(y_test, y_pred))

    print("Training completed.")

    metrics = pd.DataFrame({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }, index=model_names)

    metrics = metrics.sort_values("F1", ascending=False)
    print(f"Best performing model: {metrics.index[0]}")
    best_model = fitted_models[model_names.index(metrics.index[0])]
    y_pred = best_model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Other Models:")

    return metrics