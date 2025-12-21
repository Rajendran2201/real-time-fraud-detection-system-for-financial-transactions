from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression(random_state: int = 42):
    """
    Create a balanced Logistic Regression baseline model.
    """
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )


def get_decision_tree(random_state: int = 42):
    """
    Create a balanced Decision Tree baseline model.
    """
    return DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=random_state,
    )


def get_random_forest(random_state: int = 42):
    """
    Create a balanced Random Forest baseline model.
    """
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
