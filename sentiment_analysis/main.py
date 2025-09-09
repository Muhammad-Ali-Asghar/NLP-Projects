import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    :param file_path: Path to the dataset file.
    :return: Preprocessed DataFrame.
    """
    # Load dataset
    data = pd.read_csv(file_path)

    # Ensure the dataset has 'Text' and 'Score' columns
    if 'Text' not in data.columns or 'Score' not in data.columns:
        raise ValueError("Dataset must contain 'Text' and 'Score' columns.")

    # Map scores to binary sentiment labels
    def map_sentiment(score):
        if score in [4, 5]:
            return 1  # Positive
        elif score in [1, 2]:
            return 0  # Negative
        else:
            return None  # Neutral or ignored

    data['sentiment'] = data['Score'].apply(map_sentiment)

    # Drop rows with neutral sentiment
    data = data.dropna(subset=['sentiment'])

    # Preprocess reviews
    stop_words = set(stopwords.words('english'))
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return ' '.join(tokens)

    data['review'] = data['Text'].apply(preprocess_text)
    return data

def extract_features(data, method='tfidf'):
    """
    Convert text data to numerical format using TF-IDF or CountVectorizer.
    :param data: List of text data.
    :param method: Feature extraction method ('tfidf' or 'count').
    :return: Feature matrix.
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
    elif method == 'count':
        vectorizer = CountVectorizer()
    else:
        raise ValueError("Invalid method. Choose 'tfidf' or 'count'.")

    features = vectorizer.fit_transform(data)
    return features

def train_and_evaluate(features, labels):
    """
    Train and evaluate Logistic Regression and Naive Bayes classifiers.
    :param features: Feature matrix.
    :param labels: Target labels.
    :return: Accuracy scores for both classifiers.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_predictions)

    return lr_accuracy, nb_accuracy

if __name__ == "__main__":
    # Load and preprocess data
    file_path = "Dataset/Reviews.csv"
    data = load_and_preprocess_data(file_path)

    # Extract features using TF-IDF
    tfidf_features = extract_features(data['review'], method='tfidf')

    # Train and evaluate models
    lr_accuracy, nb_accuracy = train_and_evaluate(tfidf_features, data['sentiment'])

    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print(f"Naive Bayes Accuracy: {nb_accuracy}")
