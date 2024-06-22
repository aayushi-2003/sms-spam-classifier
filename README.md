# SMS Spam Classifier Model
This project implements a simple ML classifier model to discern between two categories: "spam" and "not spam". The model is deployed as a web application using Streamlit, allowing users to input text messages and receive predictions on whether they are spam or not spam.

App live on:
https://sms-spam-classifier-aayushi.streamlit.app/

This model is trained on dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Process
I have implemented the following process to build the model:

- Data Cleaning: Removed missing values and redundant data.
- Exploratory Data Analysis: Observed and visualized the number of characters, words, and sentences as a comparison between spam and non-spam data.
- Text Preprocessing: Removed stopwords and punctuations, and applied stemming to the remaining texts, using the NLTK library.
- Model Building: Used TF-IDF Vectorizer technique and Multinomial Naive Bayes Model for training.

The implementation details can be found in [spam detection.ipynb][1].

It is to be noted that this model is a high precision model with a precision score of 1.0 and an accuracy score of 0.97.
We require a higher precision model as compared to a higher accuracy model since we want to minimize false positives. False positives occur when legitimate messages are incorrectly classified as spam which can lead to user inconvenience or missed important messages. However we can accept some spam messages being classified as ham (i.e not having perfect accuracy) as a tradeoff for a higher precision model.

[1]: https://github.com/aayushi-2003/sms-spam-classifier/blob/main/spam%20detection.ipynb


## Requirements
1. Python3
2. nltk 
3. scikit-learn

## Installation
To run this project locally, follow these steps:

1. Clone the repository and install the necessary python libraries:
``` sh
pip install -r requirements.txt
```
2. Run the streamlit app:
``` sh
streamlit run app.py
```
## Usage
Once the Streamlit app is running, you can input text messages into the provided text area. Upon clicking the "Predict" button, the app will process the input text and display whether the message is predicted to be spam or not spam.

This is my first ML Model and would love to know about any improvements and suggestions.

Thank you!
