from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import openai

app = Flask(__name__)

# Load the data from the CSV file
data = pd.read_csv('embeddings.csv')

# The embeddings are in all but the last column
X = data.iloc[:, :-1]

# The labels are in the last column
labels = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a SVM model with RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Evaluate the model
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))

# Replace this with your actual OpenAI API key
openai.api_key = 'replace with your apu key'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    new_sentence = request.form.get('sentence', '')

    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=new_sentence
    )
    new_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1)  # Reshape the embedding to have the right dimensions
    predicted_label = model.predict(new_embedding)

    # Use the chat API to generate a response
    chat_response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "Please act as an intelligent assistant and determine the appropriate category for the following sentence. Here are the potential labels: Cash, Health, Data Modification, Resettlement,legal, Other."},
            {"role": "user", "content": new_sentence}
        ]
    )

    return jsonify({
        'predicted_label': predicted_label[0],
        'assistant_response': chat_response.choices[0].message['content'],
    })


import csv

@app.route('/feedback', methods=['POST'])
def feedback():
    new_sentence = request.form.get('sentence', '')
    was_prediction_correct = 'correct_prediction' in request.form
    was_assistant_correct = 'correct_assistant' in request.form

    # Save feedback to CSV
    with open('feedback.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([new_sentence, was_prediction_correct, was_assistant_correct])

    return jsonify({'status': 'success'})



if __name__ == "__main__":
    app.run(debug=True)
