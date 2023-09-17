import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os  # Add this import for path manipulation

from flask import Flask, render_template, request, jsonify

nltk.download('popular')

app = Flask(__name__)
app.static_folder = 'static'

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intent data from Categories.json using os.path.join
categories_file = os.path.join(os.path.dirname(__file__), 'Categories.json')
intents = json.loads(open(categories_file).read())

# Load preprocessed data, classes, and trained chatbot model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

# Function to clean up a sentence using tokenization and lemmatization
def cleanUpSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word) for word in sentenceWords]
    return sentenceWords

# Function to convert a sentence into a bag of words
def bagOfWords(sentence):
    sentenceWords = cleanUpSentence(sentence)
    bag = [0] * len(words)
    for w in sentenceWords:
        for i, word in enumerate(words):
            if w == word:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class (intent) of a sentence
def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for r in results:
        intent = {
            'intent': classes[r[0]],
            'probability': str(r[1])
        }
        returnList.append(intent)
    return returnList

# Function to get a random response based on the predicted intent
def getResponse(intentsList, intentsJson):
    if not intentsList:
        # Handle the case where no intents were predicted (empty list)
        return "We couldn't find the requetsed iformation. Don't worry we're consatntly improving our web app better to assist you."

    tag = intentsList[0]['intent']
    listOfIntents = intentsJson['categories']
    for i in listOfIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get")
def chat():
    user_message = request.args.get("msg")

    # Use your existing code for processing user input and generating responses
    ints = predictClass(user_message)
    bot_response = getResponse(ints, intents)

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(port=8000)
