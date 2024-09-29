import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

with open('data.json', 'r') as file:
    data = json.load(file)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

all_examples = []
for item in data:
    preprocessed_examples = [preprocess_text(example) for example in item['examples']]
    all_examples.extend(preprocessed_examples)
    item['preprocessed_examples'] = preprocessed_examples

vectorizer = TfidfVectorizer()
example_vectors = vectorizer.fit_transform(all_examples)

joblib.dump(vectorizer, 'vectorizer.pkl')
with open('data.json', 'w') as file:
    json.dump(data, file)

print("Model and data saved successfully.")

def find_intent(user_query):
    preprocessed_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([preprocessed_query])

    similarities = cosine_similarity(query_vector, example_vectors)
    best_match_index = similarities.argmax()

    if similarities[0][best_match_index] > 0.3:
        for item in data:
            if best_match_index < len(item['preprocessed_examples']):
                return item
            best_match_index -= len(item['preprocessed_examples'])
    return None

def chatbot(user_input):
    intent = find_intent(user_input)
    if intent:
        response = f"Detected issue: {intent['description']}\n"
        response += f"Error code: {intent['code']}\n"
        response += f"Solution: {intent['solution']}"
    else:
        response = "I'm sorry, I couldn't identify the specific issue based on your input. Could you please provide more details about the problem you're experiencing?"

    return response

if __name__ == "__main__":
    print("Auto Diagnostic Chatbot: Hello! How can I help you with your car today?")
    print("(Type 'quit' to exit)")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            print("Auto Diagnostic Chatbot: Thank you for using our service. Have a great day!")
            break

        response = chatbot(user_input)
        print("Auto Diagnostic Chatbot:", response)
        print()
