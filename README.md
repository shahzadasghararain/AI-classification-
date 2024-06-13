This repository contains a Flask web application that leverages machine learning and OpenAI's API to classify text inputs and generate intelligent responses. The key features of this application include:

1. **Data Loading and Preprocessing:**
   - Loads data from `embeddings.csv`.
   - Extracts features (embeddings) and labels.
   - Splits the dataset into training and testing sets.

2. **Model Training:**
   - Trains a Support Vector Machine (SVM) model with an RBF kernel on the training data.
   - Evaluates the model's performance on training and testing sets to ensure accuracy.

3. **OpenAI API Integration:**
   - Uses OpenAI's Embedding API to convert input sentences into embeddings.
   - Employs the OpenAI Chat API to generate responses based on input sentences and predefined categories.

4. **Flask Web Application:**
   - Provides a home route (`/`) that serves an HTML template.
   - Offers a prediction route (`/predict`) that accepts POST requests with input sentences, processes the text, predicts the category, and returns both the predicted label and an intelligent response in JSON format.

To set up and run the application:
- Ensure you have the necessary dependencies installed.
- Replace the placeholder OpenAI API key with your actual API key.
- Run the Flask application and access it via your local server.

This application demonstrates the integration of traditional machine learning models with advanced AI capabilities, providing a robust solution for text classification and interactive responses.

