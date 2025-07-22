

# 📦 Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud

# 📚 NLTK Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize lemmatizer and stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tokens = word_tokenize(text.lower())  # Lowercase + tokenize
    tokens = [t for t in tokens if t.isalpha()]  # Remove punctuation
    tokens = [t for t in tokens if t not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize
    return ' '.join(tokens)

# 📄 Step 2: Load & Preprocess Dataset
df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'Text']
df = df[['Sentiment', 'Text']]
df.drop_duplicates(subset='Text', inplace=True)
df.dropna(inplace=True)

# Encode Sentiment Labels
label_encoder = LabelEncoder()
df['Sentiment_Label'] = label_encoder.fit_transform(df['Sentiment'])

# Apply NLTK text cleaning
df['Clean_Text'] = df['Text'].apply(clean_text)

# 📊 Step 3: Visualizations - Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Sentiment')
plt.title("Sentiment Distribution")
plt.show()

# 🧾 Word Clouds for Each Sentiment
for sentiment in df['Sentiment'].unique():
    text = " ".join(df[df['Sentiment'] == sentiment]['Clean_Text'])
    wc = WordCloud(width=600, height=400, background_color='white').generate(text)
    plt.figure(figsize=(6,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"WordCloud for {sentiment}")
    plt.show()

# ✒️ Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Clean_Text'])
y = df['Sentiment_Label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🤖 Step 5: Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n📈 {name} Performance:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Test_Case:1

# 🧪 Step 6: Test Custom Sentences
test_cases = [
    "I love this product, it's amazing!",
    "Worst experience ever, never again.",
    "It's okay, not good not bad.",
    "What is this even supposed to mean?",
]

# Clean and transform test cases
clean_test = [clean_text(text) for text in test_cases]
test_vectors = vectorizer.transform(clean_test)

# Pick a model for prediction
chosen_model = models["Logistic Regression"]
predictions = chosen_model.predict(test_vectors)

print("\n🧪 Custom Test Case Results:")
for text, pred in zip(test_cases, predictions):
    print(f"'{text}' ➡ {label_encoder.inverse_transform([pred])[0]}")

# Test_Case:2

# 🧪 Extended Test Cases for Evaluation
extra_test_cases = [
    "I absolutely love the new features! Great job!",            # Positive
    "This app keeps crashing. I'm so frustrated.",               # Negative
    "Well, that was... something. Not sure how I feel.",         # Neutral/Ambiguous
    "Amazing interface but horrible customer support.",          # Mixed sentiment
    "What even is this update supposed to fix?",                 # Irrelevant/Negative
    "Cool.",                                                     # Neutral
    "Totally ruined my weekend. Thanks a lot.",                  # Sarcastic Negative
    "Nothing much to say. It's fine I guess.",                   # Neutral
    "Can someone explain what this is about?",                   # Irrelevant
    "Best update so far! Keep up the good work!",                # Positive
]

# Preprocess and predict
clean_extra = [clean_text(text) for text in extra_test_cases]
extra_vectors = vectorizer.transform(clean_extra)
extra_predictions = chosen_model.predict(extra_vectors)

print("\n🧪 Additional Test Case Results:")
for text, pred in zip(extra_test_cases, extra_predictions):
    print(f"'{text}' ➡ {label_encoder.inverse_transform([pred])[0]}")
