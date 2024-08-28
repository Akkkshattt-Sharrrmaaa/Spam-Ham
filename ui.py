import streamlit as st
import picklear
import string
from nltk import PorterStemmer
from nltk.corpus import stopwords

with open('svm_model.pkl', 'rb') as svm:
    svm_model = pickle.load(svm)

with open('lr_model.pkl', 'rb') as lr:
    lr_model = pickle.load(lr)

with open('nb_model.pkl', 'rb') as nb:
    nb_model = pickle.load(nb)

with open('count_vectorizer.pkl', 'rb') as cv:
    vectorizer = pickle.load(cv)


def process_text(text):
    words_without_punctuation = [char for char in text if char not in string.punctuation]
    words_without_punctuation = ''.join(words_without_punctuation)

    return ' '.join(
        [word for word in words_without_punctuation.split() if word.lower() not in stopwords.words('english')])


def vectorize_text(text):
    message_bagofwords = vectorizer.transform(text)
    return message_bagofwords


def stem_text(text):
    stemmer = PorterStemmer()
    return ''.join([stemmer.stem(word) for word in text])


def make_prediction(model, text):
    processed_text = process_text(text)
    stemmed_text = stem_text(processed_text)
    vectorized_text = vectorize_text([stemmed_text])

    prediction = model.predict(vectorized_text)
    return prediction


def main():

    st.title('Spam🤬 =/= Ham😃')
    st.subheader('Easily check whether a mail is  :red[spam] or  :blue[ham]', divider='orange')

    model_choice = st.radio(
        "Which model would you like to use?",
        ['Logistic Regression', 'Support Vector Classifier', 'Naive Bayes'],
        captions=["Accuracy : 99.1%", "Accuracy : 97.5% ", "Accuracy : 99.1%"]
    )

    txt = st.text_area('Enter email body',
                       "Please paste here the body of the email that you wish to analyse.")

    if st.button("Predict", type="primary"):
        if txt:
            if model_choice == 'Logistic Regression':
                prediction = make_prediction(lr_model, txt)
            elif model_choice == 'Support Vector Classifier':
                prediction = make_prediction(svm_model, txt)
            elif model_choice == 'Naive Bayes':
                prediction = make_prediction(nb_model, txt)

            if prediction == 0:
                st.success(":green[ham]")
            else:
                st.error(":red[spam]")

        else:
            st.warning("Please enter some text")


if __name__ == '__main__':
    main()
