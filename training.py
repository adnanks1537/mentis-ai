import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf  # Import TensorFlow

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intent data from Categories.json
intents = json.loads(open('Categories.json').read())

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignoreCharacters = [',', '.', '!', '?']

# Process and preprocess the intent data
for intent in intents['categories']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignore characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreCharacters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save the preprocessed data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize lists for training data
training_bags = []  # Separate list for bags
training_output = []  # Separate list for output rows
outputEmpty = [0] * len(classes)

# Create training data
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    
    training_bags.append(bag)  # Append the bag to bags
    training_output.append(outputRow)  # Append the output row to output rows

# Shuffle the training data (keeping bags and output rows synchronized)
combined_data = list(zip(training_bags, training_output))
random.shuffle(combined_data)
training_bags, training_output = zip(*combined_data)

# Convert training data to NumPy arrays
training_bags = np.array(training_bags)
training_output = np.array(training_output)

# Build a sequential neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(training_bags[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(training_output[0]), activation='softmax'))

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(training_bags, training_output, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot.h5')

print('Training Completed')
