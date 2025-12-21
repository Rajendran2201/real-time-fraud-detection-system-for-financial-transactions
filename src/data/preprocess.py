import pandas as pd
import joblib 

# preprocessing and feature engineering
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# handling imbalanced data 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


def split_features_target(df : pd.DataFrame, target_col: str):
  """
  Splits the given DataFrame into features (X) and the target (y)

  This function enforces a strict separation between features and target
  to prevent accidental leakage during preprocessing.

  Parameters
  ----------
  df : pandas.DataFrame
    The raw input data (transaction data) to be cleaned and provided for the training.
  target_col : str
    The target feature in the DataFrame. The end goal is to predict the target_col for the data sample. 

  Returns
  ----------
  X : pandas.DataFrame 
    Feature matrix.
  y : pandas.Series
    Target vector.
  """
  
  X = df.drop(columns=[target_col])
  y = df[target_col]
  return X, y


def startified_train_val_test_split(
  X: pd.DataFrame,
  y: pd.Series, 
  random_state: int, 
  test_size: float = 0.3,
  val_fraction_of_temp: float = 0.5
):
  """
  Perform a startified train/validation/test split. 

  Class proportions are preserved across all splits to endure consistent
  evaluation on highly imbalanced datasets. 

  Parameters
  ------------
  X : pandas.DataFrame
    Feature matrix. 
  y : pandas.Series
    Target vector .
  random_state : int
    Seed used to ensure reproducible splits/
  test_size : float, optional 
    Fraction of the dataset reserved for validation and test splits.
  val_fraction_of_temp : float, optional
    Fraction of the temporary split used for validation.

  Returns 
  -----------
  X_train : pandas.DataFrame
    Training features.
  X_val : pandas.DataFrame
    Validation features. 
  X_test : pandas.DataFrame
    Test features. 
  y_train : pandas.Series
    Training labels. 
  y_val : pandas.Series
    Validation labels. 
  y_test : pandas.Series
    Test labels. 
  """

  X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=test_size, 
    stratify=y, 
    random_state=random_state,
  )

  X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=val_fraction_of_temp,
    stratify=y_temp, 
    random_state=random_state,
  )

  return X_train, X_val, X_test, y_train, y_val, y_test


def scale_and_persist(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: Path,
):
  """
  Scale features using statistics learned from the training data. 

  The scaler is fit exclusively on the training set to prevent data leakage
  and is persisted for consistent use during inference. 

  Parameters
  -------------
  X_train : pandas.DataFrame
    Training feature matrix. 
  X_val : pandas.DataFrame
    Validation feature matrix. 
  X_test : pandas.DataFrame 
    Test feature matrix. 
  scaler_path : pathlib.Path 
    Path where the fitted scaler will be saved. 

  Returns
  -------------
  X_train_scaled : pandas.DataFrame
    Scaled training features
  X_val_scaled : pandas.DataFrame
    Scaled validation features. 
  X_test_scaled : pandas.DataFrame
    Scaled test features. 
  """
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  X_test_scaled = scaler.transform(X_test)

  joblib.dump(scaler, scaler_path)

  X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index
  )

  X_val_scaled = pd.DataFrame(
    X_val_scaled, columns=X_val.columns, index=X_val.index
  )

  X_test_scaled = pd.DataFrame(
    X_test_scaled, columns=X_test.columns, index=X_test.index
  )

  return X_train_scaled, X_val_scaled, X_test_scaled


def ensure_series(y: pd.Series):
  """
  Ensure target is a 1D pandas Series.

  This utility prevents shape-related issues when interfacing with 
  resampling libraries. 

  Parameters
  ------------
  y : pandas.Series
    Target variable.
  
  Returns 
  ------------
  y : pandas.Series 
    One-dimensioanl target vector 
  """

  if hasattr(y, "squeeze"):
    return y.squeeze()
  return y


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int):
  """
  Apply SMOTE oversampling to balance the training dataset. 

  Synthetic minority samples are generated based on feature-space
  similarity to improve class balance. 

  Parameters
  ------------
  X : pandas.DataFrame
    Feature matrix.
  y : pandas.Series
    Target vector. 
  random_state : int
    Seed for reproducible resampling. 

  Returns 
  ------------
  X_resampled : pandas.DataFrame
    Resampled feature matrix. 
  y_resampled : pandas.Series
    Resampled target vector. 
  """
  smote = SMOTE(random_state=random_state)
  return smote.fit_resample(X, ensure_series(y))


def apply_random_undersampling(X: pd.DataFrame, y: pd.Series, random_state: int):
  """
  Apply random undersampling to balanced the training dataset 

  Parameters
  ------------
  X : pandas.DataFrame
    Feature matrix. 
  y : pandas.Series
    Target vector. 
  random_state : int
    Seed for reproducible resampling.

  Returns
  ------------
  X_resampled : pandas.DataFrame
    Resampled feature matrix. 
  y_resampled : pandas.Series
    Resampled target vector. 
  """
  rus = RandomUnderSampler(random_state=random_state)
  return rus.fit_resample(X, ensure_series(y))

def apply_smote_tomek(X: pd.DataFrame, y: pd.Series, random_state: int):
  """
  Apply SMOTETomek to balanced the training data.

  Parameters
  --------------
  X : pandas.DataFrame
    Feature matrix. 
  y : pandas.Series
    Target vector. 
  random_state : int 
    Seed for reproducible resampling. 

   Returns 
  ------------
  X_resampled : pandas.DataFrame
    Resampled feature matrix. 
  y_resampled : pandas.Series
    Resampled target vector. 
  """
  smt = SMOTETomek(random_state=random_state)
  return smt.fit_resample(X, ensure_series(y))