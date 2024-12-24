import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time
from googleapiclient.discovery import build

# Load the emotion detection model
try:
    model = tf.keras.models.load_model("D:/Emotion-Based-Music-Recommender-main/models/emotion_model.h5")
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
    api_key = 'AIzaSyD4-X0747wo4mXqJrxHwWb7mo1Yq3JhUhE'  # Replace with your actual API key
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

# Recommend songs based on selected artist
def recommend_songs_from_artist(songs, selected_artist):
    filtered_songs = [
        f"<a href='{song['url']}' target='_blank'>{song['title']}</a>"
        for song in songs if song['artist'].lower() == selected_artist.lower()
    ]
    if not filtered_songs:
        return "No songs found for the selected artist."
    return "<br>".join(filtered_songs)

# Real-time emotion detection using webcam
def real_time_emotion_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    st.write("Detecting emotion...")

    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        # Display the current frame
        video_placeholder.image(frame, channels="BGR")

        # Preprocess the frame for emotion detection
        img_arr = preprocess_image(frame)
        prediction = model.predict(img_arr)
        emotions = ["angry", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]
        emotion = emotions[np.argmax(prediction)]

        st.session_state['emotion_detected'] = emotion
        st.success(f"Detected Emotion: {emotion}")
        cap.release()  # Release webcam after detecting emotion
        break

        time.sleep(0.1)

    cap.release()

# Streamlit app
st.title("Emotion-Based Music Recommender")
st.write("Capture your emotion using webcam, fetch artists dynamically, and recommend songs.")

# Initialize session state for emotion
if "emotion_detected" not in st.session_state:
    st.session_state['emotion_detected'] = None

# Step 1: Start emotion detection
if st.button("Start Webcam Emotion Detection"):
    real_time_emotion_detection()

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
        recommendations = recommend_songs_from_artist(st.session_state.songs, selected_artist)
        st.markdown(recommendations, unsafe_allow_html=True)
    else:
        st.info("Please select an artist to view song recommendations.")
