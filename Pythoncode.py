import cv2
import numpy as np
import pyttsx3
import time
import serial

engine = pyttsx3.init()
ser = serial.Serial("COM5", 9600, timeout=1)  # Added timeout to prevent blocking

def speak(text):
    """Uses the text-to-speech engine to say the provided text."""
    engine.say(text)
    engine.runAndWait()
    time.sleep(0.5)

def configure_engine(rate=150, volume=1.0, voice_index=0):
    """Configures the properties of the text-to-speech engine."""
    engine.setProperty('rate', rate)  # Speed of speech
    engine.setProperty('volume', volume)  # Volume level (0.0 to 1.0)
    
    voices = engine.getProperty('voices')
    if voice_index < len(voices):
        engine.setProperty('voice', voices[voice_index].id)
    else:
        print("Invalid voice index. Using default voice.")

def camera():
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return  # Stop execution if camera fails

    font = cv2.FONT_HERSHEY_PLAIN
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        frame_id += 1
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    color = colors[class_id]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, classes[class_id], (x, y + 30), font, 2, (255, 255, 255), 2)

                    object_name = classes[class_id]
                    print(object_name)
                    
                    configure_engine(rate=100, volume=1.0, voice_index=0)
                    speak(object_name)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Loop
while True:
        data = ser.readline().strip().decode("utf-8")
        print(data)
        if data == "1":
          camera()
