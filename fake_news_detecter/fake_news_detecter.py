import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
fake_path = os.path.join('Dataset', 'Fake.csv')
true_path = os.path.join('Dataset', 'True.csv')

fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine datasets
data = pd.concat([fake, true]).reset_index(drop=True)

# Preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenize
    tokens = word_tokenize(text.lower())

    # Remove stopwords and lemmatize
    processed = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(processed)

data['processed_content'] = data['text'].apply(preprocess_text)

# Visualize common terms
fake_text = ' '.join(data[data['label'] == 0]['processed_content'])
true_text = ' '.join(data[data['label'] == 1]['processed_content'])

fake_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(fake_text)
true_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(true_text)

plt.figure(figsize=(10, 5))
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Terms in Fake News')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(true_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Terms in Real News')
plt.show()

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_content'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')

