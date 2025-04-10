# Import necessary packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from paddleocr import PaddleOCR
import pymongo
from datetime import datetime

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the webcam"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Object Detection class in a separate thread
class ObjectDetectionThread(Thread):
    def __init__(self, interpreter, input_details, output_details, input_data, imW, imH, labels, min_conf_threshold):
        Thread.__init__(self)
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        self.input_data = input_data
        self.imW = imW
        self.imH = imH
        self.labels = labels
        self.min_conf_threshold = min_conf_threshold
        self.results = []

    def run(self):
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        for i in range(len(scores)):
            if (scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * self.imH)))
                xmin = int(max(1, (boxes[i][1] * self.imW)))
                ymax = int(min(self.imH, (boxes[i][2] * self.imH)))
                xmax = int(min(self.imW, (boxes[i][3] * self.imW)))

                self.results.append(((xmin, ymin, xmax, ymax), self.labels[int(classes[i])], int(scores[i] * 100)))

# Text recognition class using PaddleOCR for OCR
class TextRecognitionThread(Thread):
    def __init__(self, frame, bbox, ocr):
        Thread.__init__(self)
        self.frame = frame
        self.bbox = bbox
        self.ocr = ocr

    def run(self):
        # Crop the frame to the bounding box area
        (xmin, ymin, xmax, ymax) = self.bbox
        cropped_frame = self.frame[ymin:ymax, xmin:xmax]

        # Perform OCR on the cropped area
        results = self.ocr.ocr(cropped_frame, rec=True)
        detected_text = ''.join([line[1][0] for line in results[0]]) if results and results[0] else ""
        print(f"Detected text: {detected_text}")
        check_car_plate_and_update_status(detected_text)
        
        if detected_text.strip():
            time.sleep(5)

        if detected_text.strip():  # If any text is detected
            cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(self.frame, detected_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# MongoDB Configuration for carplates and status
MONGO_URI = "mongodb+srv://zabarose02:akubudakhensem@cluster0.db6et.mongodb.net/"
DATABASE_NAME = "ParkingSystem"
CARPLATES_COLLECTION = "carplates"
STATUS_COLLECTION = "Status"
LOGS_COLLECTION = "Logs"


# Initialize MongoDB client
client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
carplates_collection = db[CARPLATES_COLLECTION]
status_collection = db[STATUS_COLLECTION]
logs_collection = db[LOGS_COLLECTION]


# Function to check car plate, update poll status, and log successful detections
def check_car_plate_and_update_status(detected_plate):
    try:
        # Fetch all car plates and their corresponding owner names
        carplates = carplates_collection.find({}, {"Plate": 1, "Name": 1, "_id": 0})
        carplates_dict = {entry['Plate']: entry['Name'] for entry in carplates}

        # Check if detected plate matches any in the collection
        if detected_plate in carplates_dict:
            owner_name = carplates_dict[detected_plate]
            print(f"Plate {detected_plate} recognized. Owner: {owner_name}. Updating status to true.")

            # Update the status to true
            status_collection.update_one({}, {"$set": {"status": True}})

            # Log the successful detection in Logs collection
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "Plate": detected_plate,
                "Name": owner_name
            }
            logs_collection.insert_one(log_entry)
            print(f"Log entry added: {log_entry}")

            # Wait for 10 seconds
            time.sleep(10)

            # Update the status back to false
            print(f"Resetting status to false.")
            status_collection.update_one({}, {"$set": {"status": False}})
        else:
            print(f"Plate {detected_plate} not recognized in database.")

    except Exception as e:
        print(f"Error occurred: {e}")


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True)
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.5)
parser.add_argument('--resolution', default='1280x720')
parser.add_argument('--edgetpu', action='store_true')

args = parser.parse_args()

# Model settings
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Load TensorFlow Lite model and allocate tensors
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize PaddleOCR
ocr = PaddleOCR()

videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

while True:
    frame1 = videostream.read()

    frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Object Detection in a separate thread
    detection_thread = ObjectDetectionThread(interpreter, input_details, output_details, input_data, imW, imH, labels, min_conf_threshold)
    detection_thread.start()

    # Wait for the detection thread to complete
    detection_thread.join()

    # Perform OCR for each detected object
    for result in detection_thread.results:
        ((xmin, ymin, xmax, ymax), object_name, score) = result
        cv2.rectangle(frame1, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        label = f'{object_name}: {score}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(frame1, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame1, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Create a thread for OCR only on detected objects
        text_recognition_thread = TextRecognitionThread(frame1, (xmin, ymin, xmax, ymax), ocr)
        text_recognition_thread.start()
        text_recognition_thread.join()  # Wait for OCR thread to complete

    cv2.imshow('Object detector', frame1)

    if cv2.waitKey(1) == ord('q'):
        break

videostream.stop()
cv2.destroyAllWindows()
