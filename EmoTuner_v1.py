import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from googleapiclient.discovery import build
import cv2

# Load the emotion detection model
model = tf.keras.models.load_model("D:/Emotion-Based-Music-Recommender-main/models/emotion_model.h5")

# Preprocess the image for the emotion model
def preprocess_image(image):
    img = Image.fromarray(image)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to match model's input size
    img_rgb = Image.new('RGB', img.size)  # Create an RGB image
    img_rgb.paste(img)  # Paste grayscale image onto all three channels
    img_arr = np.array(img_rgb).reshape(1, 64, 64, 3) / 255.0  # Normalize and reshape
    return img_arr

# Fetch songs dynamically using YouTube API
def fetch_songs_from_youtube(emotion):
    api_key = 'AIzaSyD4-X0747wo4mXqJrxHwWb7mo1Yq3JhUhE'  # Replace with your actual API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    search_query = f"{emotion} songs"
    request = youtube.search().list(
        q=search_query,
        part="snippet",
        maxResults=5,
        type="video"
    )
    response = request.execute()

    songs = []
    for item in response['items']:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        songs.append(f"[{title}]({video_url})")

    return songs

# Predict emotion and recommend music
def recommend_music_from_image(image):
    image_arr = preprocess_image(image)
    prediction = model.predict(image_arr)
    emotions = ["angry", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]
    emotion = emotions[np.argmax(prediction)]

    # Fetch songs dynamically based on the detected emotion
    songs = fetch_songs_from_youtube(emotion)

    return emotion, songs

# Streamlit UI
st.title("Music Recommendation Based on Emotion")
st.write("Click the button to capture an image using your webcam.")

if st.button("Capture Image"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Error: Could not capture an image.")
        else:
            # Convert BGR (OpenCV format) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display captured image
            st.image(frame, caption="Captured Image", use_column_width=True)

            # Predict and recommend
            emotion, songs = recommend_music_from_image(frame)

            st.subheader(f"Detected Emotion: {emotion}")
            st.subheader("Recommended Songs:")
            for song in songs:
                st.markdown(song, unsafe_allow_html=True)
