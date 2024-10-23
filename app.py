import streamlit as st
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from pytube import YouTube

# Load your trained model
model = load_model("sentiment.h5")  # Replace 'your_model.h5' with the actual path to your trained model file

# Load the CountVectorizer's vocabulary using pickle
with open('vectorizer_vocabulary.pkl', 'rb') as vocab_file:
    vocabulary = pickle.load(vocab_file)

# Load the CountVectorizer with the correct vocabulary
vectorizer = CountVectorizer(decode_error="replace", vocabulary=vocabulary)

# Define sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]

# Create a Streamlit app
st.title("Sentiment Analysis App")

# Create a text input box for user input
video_link = st.text_input("Enter a YouTube Video Link:")

# Create a button to trigger comment retrieval and sentiment prediction
predict_button = st.button("Fetch Comments and Predict Sentiment")

if predict_button:
    if video_link:
        try:
            yt = YouTube(video_link)
            video_id = yt.video_id
        except Exception as e:
            st.warning("Invalid YouTube video link. Please enter a valid link.")
            video_id = None

        if video_id:
            # Fetch comments using the YouTube Data API (you need to provide your own API key)
            api_key = "AIzaSyAskP8XGk-kOuiSWowfEl3EsokF378K-vw"
            url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=5"
            response = requests.get(url)

            if response.status_code == 200:
                comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response.json()["items"]]
                if comments:
                    

                    st.write("Sentiment Analysis:")
                    for comment in comments:
                        # Tokenize and vectorize the user input
                        user_input_vectorized = vectorizer.transform([comment]).toarray()

                        # Make predictions with your model
                        predictions = model.predict(user_input_vectorized)

                        # Interpret predictions
                        predicted_label = sentiment_labels[np.argmax(predictions)]

                        with st.container():
                            st.text(f"Comment: {comment}")
                            st.text(f"Predicted Sentiment: {predicted_label}")
                else:
                    st.warning("No comments found for the provided video.")
            else:
                st.warning("Failed to fetch comments. Please check the video link and your API key.")
    else:
        st.warning("Please enter a YouTube Video Link.")
