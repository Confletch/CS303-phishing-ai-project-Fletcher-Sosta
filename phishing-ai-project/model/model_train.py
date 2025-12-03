# model_train.py
#
# Authors: Connor Fletcher, Frank Sosta
#
# Purpose:

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# loads in the dataset from the csv_path located in the main
def load_dataset(csv_path: str) -> pd.DataFrame:
    print(f"Loading dataset: {csv_path}")
    
    # loads the .csv into a dataframe, 
    # then detects the seperator
    # then uses the Python csv parsine engine
    df = pd.read_csv(csv_path, sep=None, engine="python")

    # this just prints column names
    # then prints the first 10 rows
    print("\nRaw Columns:", df.columns)
    print("\nFirst 10 rows of raw data:")
    print(df.head(10))

    # converts everything to lowercase to make sure theres so case sensitive issues
    df.columns = df.columns.str.lower()

    # the two columns should read as "url" and "label"
    # prints the if not found, and if found accordingly
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Cehck .csv file; must contain 'url' and 'label' columns.\n"
            f"Columns found: {list(df.columns)}"
        )

    return df # retruns the dataframe


# takes the df dataframe, and retruns a cleaned dataframe
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # "user friendly" print message to help tract the process
    print("\nCleaning dataset...")

    # removes any of the rows where the "url" or "label" is missing
    df = df.dropna(subset=["url", "label"])


    # converts labels values to strings
    # swaps everything to lowercase to ensure no issues
    # removes any extra saces before or after the label
    df["label"] = df["label"].astype(str).str.lower().str.strip()


    # lil dictionary (helps the AI determine good from bad)
    # all good URLs will be tied to a 0
    # all potentially bad URLs will be tied to a 1
    label_mapping = {
    "0": 0,
    "1": 1
}


    # apply the mappying
    df["label"] = df["label"].map(label_mapping)

    # drop/remove anything that could not be mapped
    df = df.dropna(subset=["label"])

    # makes sure the values within the label column are integers not strings
    df["label"] = df["label"].astype(int)

    # prints the number of legitimate and phishing URLs
    print("\nLabel counts after mapping:")
    print(df["label"].value_counts())

    return df # retruns the cleaned dataset


# function for splitting the data, 
# 80% for taining 
# 20% for testing
def split_data(df: pd.DataFrame):
    print("\nSplitting data into training and testing sets...")

    X = df["url"] # X is the input
    y = df["label"] # Y is the output

    # train_test_split(): this scikit-learn function splits the data into 2 parts
    # test_size=0.2: makes it so 20% of the data becomes test set
    # so the remaining 80% becomes the training set
    # random_state=42: makes it so when you rerun the script it will be the exact same
    # stratify=y: makes sure that the training and testing sets have the same ratio of phishing and legit URLs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # tells the user the amount of training and test samples there are
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:    {len(X_test)}")

    
    return X_train, X_test, y_train, y_test
    """ X_train: URLs used for training
        X_test: URLs used for testing
        y_train: labels for training
        y_test: labels for testing
    """


def vectorize_data(X_train, X_test):
    # this function takes the training and testing URL lists
    # and turns them into TF-IDF vectors
    # TF-IDF basically measures how "important" certain character patterns are
    print("\nVectorizing URLs with TF-IDF...")

    # sets up the TF-IDF vectorizer
    # analyzer="char_wb": breaks URLs into character-level n-grams
    # ngram_range=(3, 5): looks at patterns between 3 to 5 characters long
    # min_df=5: ignores VERY rare patterns that appear less than 5 times
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5
    )

    # fits the vectorizer using the training URLs
    # then transforms them into numerical vectors for the AI
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transforms the testing URLs using the SAME vectorizer rules
    X_test_tfidf = vectorizer.transform(X_test)


    # print statements to show how many features TF-IDF created
    print("Vectorization complete.")
    print("Train TF-IDF shape:", X_train_tfidf.shape)
    print("Test TF-IDF shape: ", X_test_tfidf.shape)

    return vectorizer, X_train_tfidf, X_test_tfidf


def train_model(X_train_tfidf, y_train):
    # this function trains the Logistic Regression model

    print("\nTraining Logistic Regression model...")

    # set up the model
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    model.fit(X_train_tfidf, y_train)
    print("Training finished.")

    return model


def evaluate_model(model, X_test_tfidf, y_test):
    # this tests how well the trained model performed
    # this part used the remaining 20% the test set
    print("\nEvaluating model...\n")

    # predicts the labels for the test URLs
    y_pred = model.predict(X_test_tfidf)

    # print accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # print stats
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # print confusion matrix
    # shows how many legit URLs were marked wrong, and vice-versa
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


def save_artifacts(model, vectorizer, output_dir="."):
    # this part saves the trained model and the vectorizer
    # this is so it van be resued later, without having to retarain the AI
    print("\nSaving model and vectorizer...")

    # location to save the files
    model_path = os.path.join(output_dir, "model.pkl")
    vec_path = os.path.join(output_dir, "vectorizer.pkl")

    # saves model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # save vectorizer
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # lil output to let the users know where everything got saved
    print(f"Saved model to:      {model_path}")
    print(f"Saved vectorizer to: {vec_path}")


def main():
    # UPDATE THIS PATH ACCORDINGLY FOR YOUR .csv
    # I could not make it work without hard coding the direct path
    
    csv_path = "C:/Users/confl/OneDrive/Desktop/phishing-ai-project/data/phishing_URL.csv"

    # load raw dataset
    df = load_dataset(csv_path)

    # clean the dataset
    df = clean_dataset(df)

    # since the dataset is over 99,000 URLs, we used 50,000 to train
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    # splits into seperate training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # turns the URLs into TF-IDF
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_data(X_train, X_test)

    # actually train the AI model
    model = train_model(X_train_tfidf, y_train)

    # evaluate the accuracy and print its stats
    evaluate_model(model, X_test_tfidf, y_test)

    # save everything for later use
    save_artifacts(model, vectorizer)

# finally run the entire program
if __name__ == "__main__":
    
    main()