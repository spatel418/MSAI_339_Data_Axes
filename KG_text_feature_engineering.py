'''
NOTE TO PROFESSOR:
THE CODE IN THIS FILE WAS MADE IN PART USING GENERATIVE AI
'''

'''
What this code does:
1 condense_tsv_file, combine_files, combine_claims_with_crosswalk, and combine_with_rejection_data are used to 
  make a CSV file including rows for each application published since the start of 2013 (according to the 
  patentsview pg claim text data). The most important columns here are the application number, 4 columns identifying
  whether the application got 101, 102, 103, or 112 rejections, the submission date of those rejections, and the text 
  of the first non-cancelled claim of the application as it appeared when the application was first published. 
  All of this data is complete for applications published since the start of 2013. This data also contains a column 
  for art unit but this is incomplete - it is only there for applications with rejections.
2 clean_claim_text, preprocess_and_tokenize, and preprocess_and_tokenize_string only touch the claim text column of 
  the CSV file. They convert it into a list of strings representing the tokens in that claim. To make tokens, 
  everything is made lower case, non-alphabet and non-space characters are removed, stemming is applied, and 
  stop words are removed. 
3 build_vocabulary, fit_tfidf_vectorizer, and make_tfidf_matrix are used to produce a sparse matrix (of tokens in the
  vocabulary * claims in the dataset) and an associated TfidfVectorizer which are fitted to the data in that they
  account for the frequency of the tokens in the overall dataset. Note that they are not fitted in the sense of 
  using a classification model to fit the data to a label column - this file is just meant to prepare the text data
  for that step.
4 Overall, the main useful outputs of this code are the CSV table, the TF-IDF matrix, and the TF-IDF vectorizer.
'''

import csv
import sys
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import ast
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
import joblib
import numpy as np
from scipy import sparse
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt_tab')

#This removes everything but the first claim per patent and also removes the last 2 columns
def condense_tsv_file(input_file, output_file):
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    with open(input_file, "r", encoding="utf-8", newline="") as tsv_in, \
            open(output_file, "w", encoding="utf-8", newline="") as csv_out:
        reader = csv.DictReader(tsv_in, delimiter="\t")
        writer = csv.writer(csv_out)

        writer.writerow(["pgpub_id", "claim_text"])

        current_pub_id = None
        for row in reader:
            # Pick the first claim if it's at least 30 characters long.
            # Otherwise, pick the second claim if there is 1
            # This prevents the dataset from including cancelled first claims
            if row["claim_sequence"] == "1" and len(row["claim_text"]) > 30:
                current_pub_id = row["pgpub_id"]
                writer.writerow([
                    row["pgpub_id"],
                    row["claim_text"]
                ])
            if row["claim_sequence"] == "2" and current_pub_id != row["pgpub_id"]:
                writer.writerow([
                    row["pgpub_id"],
                    row["claim_text"]
                ])

    print("file condensed to:", output_file)

def combine_files(list_of_files, output_file):
    # This combines the various files by year

    csv.field_size_limit(sys.maxsize)

    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = None
        header_written = False

        for filename in list_of_files:

            with open(filename, "r", encoding="utf-8", newline="") as infile:
                reader = csv.DictReader(infile)

                if not header_written:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True

                for row in reader:
                    writer.writerow(row)

    print("files combined to:", output_file)

def combine_claims_with_crosswalk(claims_file, crosswalk_file, output_file):
    # this joins the claim text file with the file containing app numbers (by pub number)

    csv.field_size_limit(sys.maxsize)

    crosswalk_map = {}

    with open(crosswalk_file, "r", encoding="utf-8", newline="") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")
        for row in reader:
            pgpub_id = row["pgpub_id"]
            crosswalk_map[pgpub_id] = {
                "application_id": row["application_id"],
                "patent_id": row["patent_id"]
            }

    with open(claims_file, "r", encoding="utf-8", newline="") as csv_in, \
            open(output_file, "w", encoding="utf-8", newline="") as csv_out:

        reader = csv.DictReader(csv_in)
        fieldnames = ["pgpub_id", "application_id", "patent_id", "claim_text"]
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        matched = 0

        for row in reader:
            count += 1
            pgpub_id = row["pgpub_id"]

            if pgpub_id in crosswalk_map:
                matched += 1
                merged_row = {
                    "pgpub_id": pgpub_id,
                    "application_id": crosswalk_map[pgpub_id]["application_id"],
                    "patent_id": crosswalk_map[pgpub_id]["patent_id"],
                    "claim_text": row["claim_text"]
                }
                writer.writerow(merged_row)

    print("claims and crosswalks combined to:", output_file)

def combine_with_rejection_data(claims_file, allrej_file, output_file):
    #This combines the text + crosswalk file from earlier with the rejection data file
    #merging is done by application number

    claims_cols = ["pgpub_id", "application_id", "patent_id", "claim_text"]
    rej_cols = [
        "patentApplicationNumber",
        "hasrej101", "hasrej102", "hasrej103", "hasrej112",
        "submissionDate", "groupartunitnumber"
    ]

    claims = pd.read_csv(claims_file, usecols=claims_cols, dtype=str)
    allrej = pd.read_csv(allrej_file, usecols=rej_cols, dtype=str)

    #Convert rejection columns to 0 or 1
    for col in ["hasrej101", "hasrej102", "hasrej103", "hasrej112"]:
        allrej[col] = pd.to_numeric(allrej[col], errors="coerce").fillna(0).astype(int)

    allrej = allrej.sort_values("submissionDate", na_position="last")

    agg_dict = {
        "hasrej101": "max",
        "hasrej102": "max",
        "hasrej103": "max",
        "hasrej112": "max",
        "submissionDate": "first",
        "groupartunitnumber": "first"
    }

    allrej_agg = allrej.groupby("patentApplicationNumber", as_index=False).agg(agg_dict)

    allrej_agg = allrej_agg.rename(columns={"patentApplicationNumber": "application_id"})

    # merge so that it keeps all examples even if no rejection data
    print("Merging files")
    merged = claims.merge(allrej_agg, on="application_id", how="left")

    merged.to_csv(output_file, index=False)

    print(f"Output saved to {output_file}")


def clean_claim_text(input_file, output_file):
    #This cleans the text by making it all lower case, removing extra white space, and removing
    #non-letters and non-spaces

    only_letters_regex = re.compile(r'[^a-zA-Z\s]')
    space_regex = re.compile(r'\s+')

    chunksize = 100_000
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
        if "claim_text" not in chunk.columns:
            raise ValueError("Column 'claim_text' not found in file")

        chunk["claim_text"] = (
            chunk["claim_text"]
            .astype(str)
            .str.replace(r"\n", " ", regex=True)
            .apply(lambda x: only_letters_regex.sub("", x))  # remove all non-letters/non-spaces
            .apply(lambda x: space_regex.sub(" ", x))
            .str.strip()
            .str.lower()  # make lowercase
        )

        mode = "w" if i == 0 else "a"
        header = i == 0
        chunk.to_csv(output_file, index=False, mode=mode, header=header)

        print(f"Processed chunk {i + 1}")

    print("claims cleaned!")

def preprocess_and_tokenize_string(text):
    #this is run in preprocess_and_tokenize for each string

    text = str(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed

def preprocess_and_tokenize(input_file, output_file):
    #This tokenizes the string, removes stop words, and applies stemming

    global stop_words
    stop_words = set(stopwords.words("english"))
    global stemmer
    stemmer = PorterStemmer()

    chunksize = 50_000
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
        if "claim_text" not in chunk.columns:
            raise ValueError("Column 'claim_text' not found in file")

        chunk["claim_text"] = chunk["claim_text"].apply(preprocess_and_tokenize_string)

        mode = "w" if i == 0 else "a"
        header = i == 0
        chunk.to_csv(output_file, index=False, mode=mode, header=header)
        print(f"Processed chunk {i + 1}")

    print("tokenization done!")

def build_vocabulary(input_file, output_vocab_file):
    #This builds a vocabulary based on the tokens in the claim_text column
    #it saves the vocabulary to whatever you put in output_vocab_file which should be a .json file

    vocab = set()

    reader = pd.read_csv(input_file, chunksize=100_000, usecols=["claim_text"])

    for i, chunk in enumerate(reader):
        # Convert each token list string to Python list
        chunk["claim_text"] = chunk["claim_text"].apply(ast.literal_eval)

        # Update the vocabulary set with all tokens in this chunk
        for token_list in chunk["claim_text"]:
            vocab.update(token_list)

        print(f"Processed chunk {i + 1}, vocabulary size so far: {len(vocab):,}")

    # Convert set to sorted list for stable saving
    vocab_list = sorted(vocab)

    # Save vocabulary to JSON
    with open(output_vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)

    print(f"\nVocabulary built and saved to {output_vocab_file}")
    print(f"Final vocabulary size: {len(vocab_list):,} unique tokens")

def fit_tfidf_vectorizer(input_vocab_file, input_tokenzied_claims_file, output_fitted_tfidf_vectorizer):
    #This fits a TFIDF vectorizer to the examples and saves it.
    #This manually calculates the values rather than calling .fit so this can do it in chunks
    #it applies +1 smoothing
    #This does not produce a TF-IDF matrix. That will be done later.

    # Load vocabulary which is a list
    with open(input_vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab):,} tokens")

    #create a dictionary based on the loaded vocab list
    #this is the token to index mapping
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Initialize vectorizer
    vectorizer = TfidfVectorizer(
        vocabulary=vocab_index,
        tokenizer=str.split,
        lowercase=False,
        norm="l2",
        dtype=float,
        use_idf=True,
        min_df = 10,
        max_df = 0.8
    )

    #compute document frequencies
    #vocab_index is the token to index mapping
    token_column = "claim_text"
    chunksize = 100_000
    df_counts = np.zeros(len(vocab), dtype=np.int64)
    num_docs = 0

    reader = pd.read_csv(input_tokenzied_claims_file, chunksize=chunksize, usecols=[token_column])
    for i, chunk in enumerate(reader):
        chunk[token_column] = chunk[token_column].apply(ast.literal_eval)
        for token_list in chunk[token_column]:
            num_docs += 1
            for tok in set(token_list):
                idx = vocab_index.get(tok)
                if idx is not None:
                    df_counts[idx] += 1
        print(f"Processed chunk {i + 1}")

    # Compute IDF weights manually (so I don't have to load the whole dataset at the same time)
    idf = np.log((1 + num_docs) / (1 + df_counts)) + 1.0
    vectorizer.idf_ = idf
    vectorizer._tfidf._idf_diag = sparse.spdiags(idf, diags=0, m=len(idf), n=len(idf))

    # Save the fitted vectorizer
    joblib.dump(vectorizer, output_fitted_tfidf_vectorizer)
    print(f"TF-IDF vectorizer fitted on {num_docs:,} documents and saved.")


def make_tfidf_matrix(input_vocab_file, input_tokenized_claims_file, input_tfidf_vectorizer, output_matrix_file):
    #This builds a very large sparse matrix (rows are claims and columns are tokens in the vocab)
    #This saves the matrix as a .joblib file named whatever is in output_matrix_file

    #load vocabulary
    with open(input_vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab):,} tokens")

    #load vectorizer
    vectorizer = joblib.load(input_tfidf_vectorizer)

    #create TF-IDF matrix
    chunksize = 100_000
    X_chunks = []

    reader = pd.read_csv(input_tokenized_claims_file, chunksize=chunksize, usecols=["claim_text"])

    for i, chunk in enumerate(reader):
        # Convert tokenized list to space-joined tokens
        chunk["joined"] = chunk["claim_text"].apply(lambda x: " ".join(ast.literal_eval(x)))

        # Transform to sparse TF-IDF features
        X_chunk = vectorizer.transform(chunk["joined"])
        X_chunks.append(X_chunk)
        print(f"Transformed chunk {i + 1}")

    X = vstack(X_chunks)

    #Save TF-IDF matrix
    joblib.dump(X, output_matrix_file)
    print(f"TF-IDF matrix saved. Shape: {X.shape}")


if __name__ == "__main__":

    #make csv files with only the first claims from each app
    #input_file_list = ["/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2013.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2014.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2015.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2016.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2017.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2018.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2019.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2020.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2021.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2022.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2023.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2024.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_claims_2025.tsv"]
    #output_file_list = ["/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2013_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2014_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2015_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2016_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2017_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2018_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2019_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2020_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2021_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2022_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2023_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2024_condensed.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_2025_condensed.csv"]

    #i = 0
    #while i < len(input_file_list):
    #    condense_tsv_file(input_file_list[i], output_file_list[i])
    #    i += 1

    #combine csv files for the various years into 1 csv file with more rows
    #combine_files(output_file_list, "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_combined_claims.csv")

    #combine that with the crosswalk (mainly adds a column for application number)
    #combine_claims_with_crosswalk("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_combined_claims.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/patentsview downloads/pg_granted_pgpubs_crosswalk.tsv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_joined.csv")

    #combine that with the rejection data to add more columns (such as hasrej101)
    #combine_with_rejection_data("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_joined.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/USPTO downloads/oa_allrej_2007_2025.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_with_rejections.csv")

    #clean punctuation, extra spaces, capitalization, line breaks
    #clean_claim_text("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_with_rejections.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_cleaned.csv")

    #tokenize, remove stopwords, stem
    #preprocess_and_tokenize("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_cleaned.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_tokenized.csv")
    '''
    df1 = pd.read_csv("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", nrows=20)
    print(df1)

    df2 = pd.read_csv("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/pg_claims_tokenized.csv", nrows=20)
    print(df2)

    with open("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1  # subtract 1 for header
    print(f"Total rows: {row_count}")
    '''
    #make the vocabulary
    #build_vocabulary("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/vocabulary.json")

    #make the TF-IDF vectorizer
    #fit_tfidf_vectorizer("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/vocabulary.json", "/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_vectorizer_10_8.joblib")

    #make the TF-IDF matrix
    #make_tfidf_matrix("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/vocabulary.json", "/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_vectorizer_10_8.joblib", "/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_features_10_8.joblib")

    with open("/Users/karlgadicke/Desktop/Data science USPTO data/classification/combined_df.csv", "r",
              encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1  # subtract 1 for header
    print(f"CSV file total rows: {row_count}")


    X = joblib.load("/Users/karlgadicke/Desktop/Data science USPTO data/feature_engineering/tfidf_features.joblib")
    print("TFIDF matrix:")
    print("Matrix type:", type(X))
    print("Matrix shape:", X.shape)
    print("Rows:", X.shape[0])
    print("Columns:", X.shape[1])


