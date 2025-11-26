'''
NOTE TO PROFESSOR:
THE CODE IN THIS FILE WAS MADE IN PART USING GENERATIVE AI
'''

'''
What this code does:
For now it just makes a train test split, does classification using a logistic regression model on the text features, 
saves the model, and outputs some evaluation data. 
'''


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse


#loading stuff
#replace this with the correct column for 101 / 102 / 103 / 112
label_col = "hasrej101"
labels = pd.read_csv("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_tokenized.csv", usecols=[label_col])[label_col].fillna(0).astype(int)
print("Labels loaded:", labels.shape)

X = joblib.load("tfidf_features.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
print("TF-IDF matrix shape:", X.shape)
print("Sparsity:", X.nnz / (X.shape[0] * X.shape[1]))

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=33, #prob change this each time
    stratify=labels
)

#train logistic regression model (using only the text data)
#class_weight='balanced' makes this a lot better at least with the very imbalanced dataset of 101 rejections
model = LogisticRegression(
    penalty="l2",
    solver="saga",        # best for huge sparse matrices
    max_iter=10_000,
    n_jobs=-1,
    verbose=1,
    class_weight='balanced'
)
model.fit(X_train, y_train)

#save the model
model_path = "/Users/karlgadicke/Desktop/Data science USPTO data/classification/101_prediction_model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

#evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
