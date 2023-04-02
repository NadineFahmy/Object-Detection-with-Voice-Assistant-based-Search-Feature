import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import os

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def takeCommand():
    r = sr.Recognizer()

    with sr.Microphone() as source:

        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print(e)
        print("Unable to Recognize your voice.")
        return "None"

    return query

def AI_speak(something):
    engine = pyttsx3.init()
    engine.say(something)
    engine.runAndWait()

def Camera(thing) :
    cam = cv2.VideoCapture(0)
    result, image = cam.read()
    if result:

        cv2.imshow("Image", image)
        cv2.imwrite("detect.png", image)
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

        classes = []

        with open("coco.names", "r") as f:
            classes = f.read().splitlines()
            img = cv2.imread('detect.png')

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
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

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(100, 3))

    count_array = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
            count_array.append(label)
        my_dict = {i: count_array.count(i) for i in count_array}
        if thing in my_dict:
            AI_speak('Yes')
        else:
            AI_speak("Nothing detected")

        cv2.imshow('Image', img)
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    clear = lambda: os.system('cls')

    clear()

    while True:
        query = takeCommand().lower()
        n = ''

        if 'search' in query:
            AI_speak("What are you searching for?")
            n = takeCommand()
            Camera(n)
            break