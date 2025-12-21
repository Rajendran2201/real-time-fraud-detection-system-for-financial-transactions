from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def compute_scale_pos_weight(y):
    """
    Compute scale_pos_weight for imbalanced binary classification.

    Parameters
    ----------
    y : array-like
        Binary target vector.

    Returns
    -------
    float
        Ratio of negative to positive samples.
    """
    positives = y.sum()
    negatives = len(y) - positives
    return negatives / positives


def build_xgboost(y, random_state: int = 42):
    """
    Create an XGBoost classifier configured for imbalanced data.
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=compute_scale_pos_weight(y),
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )


def build_lightgbm(random_state: int = 42):
    """
    Create a LightGBM classifier with balanced class weights.
    """
    return LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state,
    )


def build_catboost(y, random_state: int = 42):
    """
    Create a CatBoost classifier configured for imbalanced data.
    """
    pos_weight = (len(y) - y.sum()) / y.sum()

    return CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        class_weights=[1, pos_weight],
        verbose=0,
        random_seed=random_state,
    )
