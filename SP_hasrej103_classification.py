import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix, issparse
#import numpy as np
from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TARGET_COLUMN = "hasrej103_x"
FEATURES_JOBLIB = "/content/drive/MyDrive/MSAI 339 - Data Science/tfidf_features.joblib"
CSV_PATH = "/content/drive/MyDrive/MSAI 339 - Data Science/combined_df.csv"

def prepare_data(df_claims, target_col="hasrej103_x"):
    """
    1. Creates a binary rejection column ('reject_hasrej103').
    2. Splits data into 70% Train, 15% Validation, 15% Test (stratified).
    3. Downsamples the majority class in the training set to achieve a 50/50 balance.
    4. Ensures the validation and test sets mimic the original class distribution.

    :param df: The input DataFrame containing the target column.
    :param target_col: The column to use for the target label (e.g., "hasrej103").
    :return: Indices and labels for the balanced training set, and the real-world val/test sets.
    """

    # 1. Select relevant column and create the binary target
    #df_claims = df[[target_col]].copy()

    # Create the new binary column: reject_hasrej103 (1 if >= 1, else 0)
    df_claims["reject_hasrej103"] = (df_claims[target_col].fillna(0) >= 1).astype(int)

    # Separate index (X) and the new target label (y)
    X = df_claims.index.to_series() # Use index as 'X' for splitting
    y = df_claims["reject_hasrej103"]

    print("--- Initial Data Status ---")
    print(f"Total Samples: {len(df_claims)}")
    print(f"Class Distribution:")

    # 2. Split into Train (70%) and Temp (30%), stratified
    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Split Temp (30%) into Validation (15% of total) and Test (15% of total), stratified
    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Function to print information
    def print_split_info(name, y_data):
        count = len(y_data)
        # Recalculate distribution directly from the balanced/split labels
        dist = y_data.value_counts(normalize=True).mul(100).round(2)
        # Use .get() for safety
        class_1_perc = dist.get(1, 0.0) 
        class_0_perc = dist.get(0, 0.0)
        print(f"**{name}** (N={count}): {class_1_perc}% Class 1 | {class_0_perc}% Class 0")

    #print_split_info("Train (Balanced)", y_train_balanced, train_indices_balanced)
    # The val/test sets will still show the real-world distribution
    print_split_info("Train", y_train)
    print_split_info("Validation (Real-World)", y_val)
    print_split_info("Test (Real-World)", y_test)
    
    return X_train_idx, X_val_idx.index, X_test_idx.index, y_train, y_val, y_test 

def process_features():
    print("\n--- Loading Sparse TF-IDF Features ---")
    try:
        X_all = joblib.load(FEATURES_JOBLIB)
    except Exception as e:
        print(f"Error loading features from {FEATURES_JOBLIB}: {e}")
        # Exit if features can't be loaded, as the classifier requires them
        exit()

    # Ensure X is a sparse matrix
    if not issparse(X_all):
        X_all = csr_matrix(X_all)
        print("Features converted to sparse matrix.")

    print(f"Sparse matrix shape loaded: {X_all.shape}")

    # Map the indices to the sparse feature matrix
    X_train = X_all[train_indices_b, :]
    X_val = X_all[val_indices, :]
    X_test = X_all[test_indices, :]

    return X_train, X_val, X_test

def run_xgb_classification(dataset_dict):
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
        #scale_pos_weight=scale_pos_weight_value
    ) 
    X_train, y_train_b = dataset_dict["train"] 
    X_val, y_val = dataset_dict["val"]
    X_test, y_test = dataset_dict["test"]
    print('XGBOOST CLASSIFICATION')
    model.fit(X_train, y_train_b)
    predictions_train = model.predict(X_train)
    print('---TRAIN-----')
    print(classification_report(y_train_b, predictions_train))
    print('---VAL---')
    predictions_val = model.predict(X_val)
    print(classification_report(y_val, predictions_val))
    print('---TEST---')
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    return model

def run_nb_classification(dataset_dict):
    model = MultinomialNB()
    X_train, y_train_b = dataset_dict["train"] 
    X_val, y_val = dataset_dict["val"]
    X_test, y_test = dataset_dict["test"]
    print('NAIVE BAYES CLASSIFICATION')
    model.fit(X_train, y_train_b)
    predictions_train = model.predict(X_train)
    print('---TRAIN-----')
    print(classification_report(y_train_b, predictions_train))
    print('---VAL---')
    predictions_val = model.predict(X_val)
    print(classification_report(y_val, predictions_val))
    print('---TEST---')
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    return model

    

if __name__ == "__main__":
    TARGET_COLUMN = "hasrej103_x"
    FEATURES_JOBLIB = "/content/drive/MyDrive/MSAI 339 - Data Science/tfidf_features.joblib"
    CSV_PATH = "/content/drive/MyDrive/MSAI 339 - Data Science/combined_df.csv"
    #substitute the parent path
    df = pd.read_csv(CSV_PATH)
    df = df.reset_index(drop=True)  
    df_claims = df[["claim_text", "hasrej101_x", "hasrej102_x", "hasrej103_x", "hasrej112_x"]]
    train_indices_b, val_indices, test_indices, y_train_b, y_val, y_test = prepare_data(
        df_claims,
        target_col=TARGET_COLUMN
    )
    X_train, X_val, X_test = process_features()
    dataset_dict = {
                        "train": (X_train, y_train_b),
                        "val": (X_val, y_val),
                        "test": (X_test, y_test)
                    }
    run_xgb_classification(dataset_dict)
    run_nb_classification(dataset_dict)
