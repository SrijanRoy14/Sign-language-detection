import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import tensorflow as tf

# Loading the pre-trained model
action_model = tf.keras.models.load_model('action.h5')

# Initializing mediapipe holistic model and drawing utils
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Defining the actions array
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Function to perform mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw landmarks on the image
def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(110, 256, 121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to extract keypoints from the results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    return np.concatenate([pose, face, lh, rh])

# State variables
sequence = []
sentence = []
predictions = []
threshold = 0.95

def update_frame():
    global sequence, sentence,predictions

    #print(sequence,sentence)
    ret, frame = cap.read()
    if not ret:
        vid.after(10, update_frame)
        return

    image, results = mediapipe_detection(frame, holistic)
    draw_landmarks(image, results)

    keypoints = extract_keypoints(results)
    sequence.insert(0,keypoints)
    sequence=sequence[:30]  # Keeping only the last 30 keypoints

    if len(sequence) == 30:
        res = action_model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(res)
        predictions = predictions[-10:]  # Keeping the last 10 predictions

        avg_res = np.mean(predictions, axis=0)
        if avg_res[np.argmax(avg_res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(avg_res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(avg_res)])
            else:
                sentence.append(actions[np.argmax(avg_res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.squeeze(img)
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=imgarr)
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    
    vid.after(10, update_frame)

# Creating the main application window
app = tk.Tk()
app.geometry("600x600")
app.title("ASL Converter")

# Creating the video frame and label
vidFrame = tk.Frame(app, height=400, width=600)
vidFrame.pack()
vid = tk.Label(vidFrame)
vid.pack()

# Initializing the video capture
cap = cv2.VideoCapture(0)

# Starting the holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

button = tk.Button(app, 
                   text="Start", 
                   command=update_frame,
                   activebackground="blue", 
                   activeforeground="white",
                   anchor="center",
                   bd=1,
                   bg="teal",
                   cursor="hand2",
                   disabledforeground="gray",
                   fg="black",
                   font=("Arial",14),
                   height=2,
                   
                   justify="center",
                   overrelief="raised",
                   padx=20,
                   pady=10,
                   width=15
                   )


button.pack(padx=80, pady=200)


# Running the main application loop
app.mainloop()

# Releasing resources when done
cap.release()
holistic.close()
cv2.destroyAllWindows()
