import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from googleapiclient.discovery import build

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
        maxResults=10,
        type="video"
    )
    response = request.execute()

    songs = []
    for item in response['items']:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        songs.append({"title": title, "url": video_url})

    return songs

# Initialize state variables
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
            placeholder.empty()

            # Preprocess and predict the emotion from the image
            img_arr = preprocess_image(image)
            prediction = model.predict(img_arr)
            emotions = ["angry", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]
            detected_emotion = emotions[np.argmax(prediction)]

            st.success(f"Detected Emotion: {detected_emotion}")
            st.session_state['emotion_detected'] = detected_emotion
            st.session_state["process_completed"] = True
        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Step 2: Recommend Songs
if st.session_state.get("emotion_detected"):
    st.subheader(f"Emotion Detected: {st.session_state.emotion_detected}")

    if "songs" not in st.session_state:
        songs = fetch_songs_from_youtube(st.session_state.emotion_detected)
        st.session_state.songs = songs

    st.subheader("Recommended Songs:")
    recommendations = [
        f"<a href='{song['url']}' target='_blank'>{song['title']}</a>"
        for song in st.session_state.songs
    ]
    if not recommendations:
        st.info("No songs found for the detected emotion.")
    else:
        st.markdown("<br>".join(recommendations), unsafe_allow_html=True)

# Step 3: Retake the image
if st.button("Retake Image"):
    st.session_state["process_completed"] = False
    st.session_state.pop("emotion_detected", None)
    st.session_state.pop("songs", None)
    placeholder.empty()
