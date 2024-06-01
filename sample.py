import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('asl_model.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Function to preprocess hand images for the model
def preprocess_image(image, hand_landmarks):
    h, w, _ = image.shape
    hand_rect = cv2.boundingRect(np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]))
    x, y, w, h = hand_rect
    
    # Check if the ROI is valid
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        return None
    
    hand_img = image[y:y+h, x:x+w]
    
    # Check if the hand_img is not empty
    if hand_img.size == 0:
        return None
    
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    hand_img = cv2.resize(hand_img, (28, 28))
    hand_img = hand_img / 255.0
    hand_img = np.expand_dims(hand_img, axis=-1)
    hand_img = np.expand_dims(hand_img, axis=0)
    return hand_img

# Function to classify hand image
def classify_hand_image(image, hand_landmarks):
    input_data = preprocess_image(image, hand_landmarks)
    
    if input_data is None:
        return None
    
    predictions = model.predict(input_data)
    
    predicted_label = np.argmax(predictions)
    return predicted_label

# Streamlit app
st.title("ASL Fingerspelling Recognition")
st.subheader("Using PC Camera")

# Video capture from webcam
cap = cv2.VideoCapture(0)

# Correct ASL letters mapping based on the provided image
asl_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Streamlit video display
stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Classify the hand shape
            predicted_label = classify_hand_image(rgb_frame, hand_landmarks)
            
            if predicted_label is not None:
                # Ensure the predicted label is within the valid range
                if 0 <= predicted_label < len(asl_letters):
                    predicted_letter = asl_letters[predicted_label]
                else:
                    predicted_letter = '?'
                
                # Get coordinates of the hand landmarks for display
                h, w, _ = frame.shape
                coords = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]
                
                # Display the predicted letter and hand coordinates
                cv2.putText(frame, f'Letter: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                for coord in coords:
                    cv2.circle(frame, coord, 5, (0, 255, 0), -1)
    
    # Display the frame in Streamlit
    stframe.image(frame, channels="BGR")

cap.release()
