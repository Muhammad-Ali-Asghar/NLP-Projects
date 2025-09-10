import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import tensorflow as tf

# Limit TensorFlow GPU memory usage (if GPU is available)
if tf.config.list_physical_devices('GPU'):
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
train_data_path = 'Dataset/train.csv'
test_data_path = 'Dataset/test.csv'

# Load train and test datasets
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Combine datasets for preprocessing
train_data['is_train'] = 1
test_data['is_train'] = 0
data = pd.concat([train_data, test_data], ignore_index=True)

# Preprocessing
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import re
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase, remove special characters, tokenize, remove stopwords, and lemmatize
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
data['processed_text'] = data['Description'].apply(preprocess_text)

# Separate train and test datasets after preprocessing
train_data = data[data['is_train'] == 1].drop(columns=['is_train'])
test_data = data[data['is_train'] == 0].drop(columns=['is_train'])

# Visualize most frequent words per category
def visualize_words(data, category_column, text_column):
    categories = data[category_column].unique()
    for category in categories:
        category_text = ' '.join(data[data[category_column] == category][text_column])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(category_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Most Frequent Words in {category}')
        plt.axis('off')
        plt.show()

# Uncomment to visualize
# visualize_words(data, 'category', 'processed_text')

# Vectorization
vectorizer = TfidfVectorizer(max_features=2000)

# Use sparse matrices for vectorized data
X_train = vectorizer.fit_transform(train_data['processed_text'])
y_train = train_data['Class Index']
X_test = vectorizer.transform(test_data['processed_text'])
y_test = test_data['Class Index']

# Train-test split (if needed for validation within train set)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_split, y_train_split)
lr_preds = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_split, y_train_split)
rf_preds = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train_split, y_train_split)
svm_preds = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))

# Simplify Neural Network
# Update num_classes 
num_classes = y_train_split.max() + 1

# Update the one-hot encoding 
y_train_nn = to_categorical(y_train_split, num_classes)
y_val_nn = to_categorical(y_val_split, num_classes)
# Model not trained due to insufficient RAM
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_split.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reduce batch size and epochs
nn_model.fit(
    X_train_split, y_train_nn, epochs=5, batch_size=16, validation_data=(X_val_split, y_val_nn)
)

nn_preds = nn_model.predict(X_test)
nn_preds_classes = np.argmax(nn_preds, axis=1)
print("Neural Network Accuracy:", accuracy_score(y_test, nn_preds_classes))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, nn_preds_classes))
