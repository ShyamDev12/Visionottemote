import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import webbrowser

# Load model and labels
model = tf.keras.models.load_model("model.h5")
label = np.load("labels.npy")

# MediaPipe setup
holistic = mp.solutions.holistic
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Streamlit UI setup
st.header("Fun Zone ! Capture your Emote !😁🙁😒😏")

# Session state for managing emotion detection
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

# Function to process image and predict emotion
def process_image(image):
    res = holis.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    lst = []

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)
        pred = label[np.argmax(model.predict(lst))]
        return res, pred
    return res, ""

lang = st.text_input("Language")
singer = st.text_input("Cast")

# Upload button for capturing an image
uploaded_file = st.file_uploader("Capture your image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    res, emotion = process_image(image)

    if res.face_landmarks:
        drawing.draw_landmarks(image, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
    if res.left_hand_landmarks:
        drawing.draw_landmarks(image, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        drawing.draw_landmarks(image, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    st.image(image, channels="BGR")
    st.session_state["emotion"] = emotion

    if emotion:
        st.success(f"Emotion detected: {emotion}")

btn = st.button("Recommend me movies")

if btn:
    if not st.session_state["emotion"]:
        st.warning("Please upload an image to capture your emotion first")
    else:
        emotion = st.session_state["emotion"]
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+movie+{singer}")
        st.session_state["emotion"] = ""  # Reset emotion after recommendation
