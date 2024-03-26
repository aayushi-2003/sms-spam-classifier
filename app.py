import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    finaltext = []
    for i in text:
        if i.isalnum():
            finaltext.append(i)

    text = finaltext[:]
    finaltext.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            finaltext.append(i)

    text = finaltext[:]
    finaltext.clear()

    for i in text:
        finaltext.append(ps.stem(i))

    return " ".join(finaltext)


tfidf = pickle.load(open('vectorization.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
