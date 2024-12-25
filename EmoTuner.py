import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import time
from googleapiclient.discovery import build

# Load the emotion detection model
model_path = "models/emotion_model.h5"
if not os.path.exists(model_path):
    st.error("Model file not found. Please upload the correct model file.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Preprocess the image for the emotion model
def preprocess_image(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = img.convert('L')
    img = img.resize((64, 64))
    img_rgb = Image.new('RGB', img.size)
    img_rgb.paste(img)
    img_arr = np.array(img_rgb).reshape(1, 64, 64, 3) / 255.0
    return img_arr

# Fetch songs dynamically using YouTube API
def fetch_songs_and_artists_from_youtube(emotion):
    api_key = st.secrets["YOUTUBE_API_KEY"]  # Use environment variable for API key
    if not api_key:
        st.error("YouTube API key not configured.")
        return [], {}

    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        search_query = f"{emotion} songs"
        request = youtube.search().list(
            q=search_query,
            part="snippet",
            maxResults=10,
            type="video"
        )
        response = request.execute()

        songs = []
        artist_count = {}
        for item in response['items']:
            title = item['snippet']['title']
            video_id = item['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            description = item['snippet']['description']

            # Extract artist's name
            artist = "Unknown Artist"
            if '-' in title:
                artist = title.split('-')[0].strip()
            elif 'by' in description.lower():
                artist = description.lower().split('by')[-1].strip().split('\n')[0].strip()

            if artist not in artist_count:
                artist_count[artist] = 0
            artist_count[artist] += 1

            songs.append({'title': title, 'url': video_url, 'artist': artist})

        return songs, artist_count
    except Exception as e:
        st.error(f"Error fetching data from YouTube: {e}")
        return [], {}

# Streamlit app
st.title("Emotion-Based Music Recommender")
st.write("Upload your image, detect your emotion, fetch artists dynamically, and recommend songs.")

# Step 1: Manage state for image capture and retake functionality
if "process_completed" not in st.session_state:
    st.session_state["process_completed"] = False

# Placeholder for camera input
placeholder = st.empty()

if not st.session_state["process_completed"]:
    uploaded_image = placeholder.camera_input("Capture an image for emotion detection")

    if uploaded_image:
        try:
            # Convert the captured image to a format that can be processed
            image = np.array(Image.open(uploaded_image))

            # Clear the image placeholder after capture
            placeholder.empty()  # This hides the camera input and the captured image

            # Preprocess and predict the emotion from the image
            img_arr = preprocess_image(image)
            prediction = model.predict(img_arr)
            emotions = ["angry", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]
            detected_emotion = emotions[np.argmax(prediction)]

            st.success(f"Detected Emotion: {detected_emotion}")
            st.session_state['emotion_detected'] = detected_emotion
            st.session_state["process_completed"] = True  # Mark process as completed
        except Exception as e:
            st.error(f"Error processing the image: {e}")


# Step 2: Fetch Artists
if st.session_state.get("emotion_detected"):
    st.subheader(f"Emotion Detected: {st.session_state.emotion_detected}")
    if "artists" not in st.session_state:
        songs, artist_count = fetch_songs_and_artists_from_youtube(st.session_state.emotion_detected)
        st.session_state.songs = songs
        st.session_state.artists = [artist for artist in artist_count.keys()]

    selected_artist = st.selectbox("Select an Artist", ["Select an Artist"] + st.session_state.artists, key="artist_selection")

    # Step 3: Recommend Songs (only if an artist is selected)
    if selected_artist != "Select an Artist" and selected_artist:
        st.subheader(f"Songs by {selected_artist}")
        recommendations = [
            f"<a href='{song['url']}' target='_blank'>{song['title']}</a>"
            for song in st.session_state.songs if song['artist'].lower() == selected_artist.lower()
        ]
        if not recommendations:
            st.info("No songs found for the selected artist.")
        else:
            st.markdown("<br>".join(recommendations), unsafe_allow_html=True)
    else:
        st.info("Please select an artist to view song recommendations.")
# Step 2: Retake the image after the process is done
if st.session_state["process_completed"]:
    st.subheader(f"Emotion Detected: {st.session_state.get('emotion_detected', 'N/A')}")

    # Button to retake the image
    if st.button("Retake Image"):
        st.session_state["process_completed"] = False
        st.session_state.pop("emotion_detected", None)  # Reset detected emotion
        placeholder.empty()  # Clear any leftover UI elements
