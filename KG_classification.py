'''
NOTE TO PROFESSOR:
THE CODE IN THIS FILE WAS MADE IN PART USING GENERATIVE AI
'''

'''
What this code does:
Classification using a logistic regression model, KNN with dimensionality reduction, or neural net with dimensionality
reduction. For each of these it then evaluates the model on the test set and outputs some data on how good the model is
'''

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def logistic_regression_classifier(label_csv, label_col, features_joblib, vectorizer_joblib, save_model_location):
    """
    :param label_csv:
    :param label_col: note that this currently only works on columns that are 0 or 1. It fills nothing with 0.
    :param features_joblib:
    :param vectorizer_joblib:
    :param save_model_location:
    :return:
    """

    #loading stuff
    labels = pd.read_csv(label_csv, usecols=[label_col])[label_col].fillna(0).astype(int)
    print("Labels loaded:", labels.shape)

    X = joblib.load(features_joblib)
    vectorizer = joblib.load(vectorizer_joblib)
    print("TF-IDF matrix shape:", X.shape)
    print("Sparsity:", X.nnz / (X.shape[0] * X.shape[1]))

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=33,
        stratify=labels
    )

    #train logistic regression model (using only the text data)
    #class_weight='balanced' makes this a lot better at least with the very imbalanced dataset of 101 rejections
    model = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=10_000,
        n_jobs=-1,
        verbose=1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    #save the model
    joblib.dump(model, save_model_location)
    print(f"Model saved to: {save_model_location}")

    #evaluate the model
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

def knn_with_dimensionality_reduction(label_col, k):
    #KNN seems like a bad idea for a dataset this huge but seeing if it works if I do dimensionality reduction

    #load stuff
    tfidf_path = "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_features.joblib"
    label_csv_path = "/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv"
    X = joblib.load(tfidf_path)
    labels = pd.read_csv(label_csv_path, usecols=[label_col])[label_col].fillna(0).astype(int)
    print("done loading stuff")

    #make train test split
    #im using 1000 test size because I think this will take a while for each test because KNN is inefficient with this many samples
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=2000,
        stratify=labels,
        random_state=34
    )
    print("done making train test split")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    #dimensionality reduction
    num_dimensions = 100
    print(f"Performing TruncatedSVD with {num_dimensions} components...")
    svd = TruncatedSVD(
        n_components=num_dimensions,
        n_iter=7,
        random_state=70
    )
    X_train_svd_transformed = svd.fit_transform(X_train)
    X_test_svd_transformed = svd.transform(X_test)
    X_train_svd = X_train_svd_transformed.astype("float32")
    X_test_svd = X_test_svd_transformed.astype("float32")
    print("dimensionality reduction done")
    print("Dimensionality reduced train shape:", X_train_svd.shape)
    print("Dimensionality reduced test shape:", X_test_svd.shape)

    #make KNN model
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute"
    )
    knn.fit(X_train_svd, y_train)

    #evaluate KNN model
    y_pred = knn.predict(X_test_svd)
    print(classification_report(y_test, y_pred))

#this class is for the neural net classifier and was made entirely using generative AI
class MemmapDataset(Dataset):
    def __init__(self, data_path, label_path, n_features):
        self.X = np.memmap(data_path, dtype="float32", mode="r+")
        total_rows = self.X.size // n_features
        self.X = self.X.reshape(total_rows, n_features)

        self.y = np.fromfile(label_path, dtype="int8")

        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

#this neural net part is particularly heavy on the generative AI assistance
def neural_net_classifier_with_dimensionality_reduction():
    #load stuff
    label_col = "hasrej101_x"

    labels = pd.read_csv("combined_df.csv", usecols=[label_col])[label_col].fillna(0).astype(int)
    y = labels.to_numpy()

    X = joblib.load("tfidf_features.joblib")
    print("done loading stuff")

    #create train test split
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=67,
        shuffle=True
    )

    X_train_sparse = X[train_idx]
    X_test_sparse = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    print("done making train test split")

    #This commented out part of the code takes a while to run. I already have the output saved which is why it's
    #commented out here but someone running this for the first time will have to run this part
    '''
    #dimensionality reduction with truncated SVD
    n_components = 300

    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=7,
        random_state=90
    )
    
    
    X_train_reduced = svd.fit_transform(X_train_sparse)
    X_test_reduced = svd.transform(X_test_sparse)
    # Load previously saved memmap files
    print("dimensionality reduction done")

    #making memmap files to manage memory
    X_train_mm = np.memmap(
        "X_train_reduced.dat",
        dtype="float32",
        mode="w+",
        shape=X_train_reduced.shape
    )
    X_train_mm[:] = X_train_reduced.astype("float32")
    del X_train_reduced

    X_test_mm = np.memmap(
        "X_test_reduced.dat",
        dtype="float32",
        mode="w+",
        shape=X_test_reduced.shape
    )
    X_test_mm[:] = X_test_reduced.astype("float32")
    del X_test_reduced

    y_train.astype("int8").tofile("y_train.bin")
    y_test.astype("int8").tofile("y_test.bin")

    np.save("svd_components.npy", svd.components_)
    np.save("svd_explained_var.npy", svd.explained_variance_ratio_)
    print("done saving memmap files")
    

    #stuff for loading memmap files
    train_dataset = MemmapDataset("X_train_reduced.dat", "y_train.bin", n_components)
    test_dataset = MemmapDataset("X_test_reduced.dat", "y_test.bin", n_components)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=0)
    print("done loading memmap files")
    '''

    # Load previously saved memmap files
    train_dataset = MemmapDataset("X_train_reduced.dat", "y_train.bin", 300)
    test_dataset = MemmapDataset("X_test_reduced.dat", "y_test.bin", 300)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=0)
    print("done loading previously saved memmap files")

    #make classifier
    class ClaimClassifier(nn.Module):
        def __init__(self, input_dim=300):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
            #nn.Sigmoid() should be the last line in here when I'm not using class weights

        def forward(self, x):
            return self.model(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClaimClassifier(input_dim=300).to(device) #EDIT THIS AS WELL IF I CHANGE HUMBER OF DIMENSIONS FROM 300

    #for class weights (101 is very imbalanced toward class 0)
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()

    pos_weight_value = num_neg / num_pos
    print("pos_weight =", pos_weight_value)

    #this is for using class weights
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    #criterion = nn.BCELoss() #this is binary cross entropy and is what I use when I'm not using class weights
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #this is an alternative to regular gradient descent

    print("done making classifier")

    #train classifier
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.unsqueeze(1).to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "nn_rejection_model.pt")

    print("done training classifier")

    #evaluate classifier
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy().flatten()
            all_preds.extend(preds > 0.5)
            all_true.extend(y_batch.numpy().astype(int))

    print("Test Accuracy:", accuracy_score(all_true, all_preds))
    print("Test Precision:", precision_score(all_true, all_preds))
    print("Test Recall:", recall_score(all_true, all_preds))
    print("Test F1:", f1_score(all_true, all_preds))


if __name__ == '__main__':
    #logistic_regression_classifier("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "hasrej101_x", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_features_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_vectorizer_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/classification/101_prediction_model_10_8.joblib")
    #logistic_regression_classifier("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "hasrej102_x", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_features_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_vectorizer_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/classification/102_prediction_model_10_8.joblib")
    #logistic_regression_classifier("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "hasrej103_x", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_features_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_vectorizer_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/classification/103_prediction_model_10_8.joblib")
    #logistic_regression_classifier("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "hasrej112_x", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_features_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_vectorizer_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/classification/112_prediction_model_10_8.joblib")

    #knn_with_dimensionality_reduction("hasrej101_x", 4)
    #knn_with_dimensionality_reduction("hasrej102_x", 4)
    #knn_with_dimensionality_reduction("hasrej103_x", 4)
    #knn_with_dimensionality_reduction("hasrej112_x", 4)

    neural_net_classifier_with_dimensionality_reduction()
