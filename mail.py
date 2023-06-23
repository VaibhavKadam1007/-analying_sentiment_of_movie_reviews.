# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Step 1: Dataset Preparation
# data = pd.read_csv('sentiment_data.csv')
# texts = data['text']
# labels = data['label']

# # Step 3: Feature Extraction
# vectorizer = CountVectorizer()
# features = vectorizer.fit_transform(texts)

# # Step 4: Model Training
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)

# # Step 5: Model Evaluation
# y_pred = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Step 6: Model Deployment
# text_to_predict = ["I love this product! It exceeded my expectations."]
# features_to_predict = vectorizer.transform(text_to_predict)
# prediction = classifier.predict(features_to_predict)
# print("Sentiment Prediction:", prediction)
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

reviews = []
labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        reviews.append(movie_reviews.words(fileid))
        labels.append(category)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

reviews = [preprocess_text(' '.join(review)) for review in reviews]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

