import os
import time
from datetime import datetime
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from PyQt6.QtCore import QThread,QTimer,QObject,pyqtSignal,pyqtSlot
from .utils import *
from .tracker import FaceTracker
from pathlib import Path
from database_handler import EmotionDatabase
from pdf_utils import EmotionReport

current_dir = Path(__file__).parent
# construct the argument parser and parse the arguments
class FaceEmotionRecognation(QObject):
    emotionRsult = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob'])
    def __init__(self):
        super().__init__()
      
               
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-d", "--device",
                        default="gpu",
                        choices=["gpu", "cpu"],
                        type=str,
                        help="hardware device")

        self.ap.add_argument("-rm", "--model",
                        default="EmotionClassifier_V5",
                        type=str,
                        help="recognition model")

        self.ap.add_argument("-i", "--app_input",
                        default="0",
                        type=str,
                        help="stream input or video path")

        self.ap.add_argument("-rt", "--recognition_threshold",
                        default=0.6,
                        type=float,
                        help="threshold of recognition model")

        self.ap.add_argument("-dt", "--detection_threshold",
                        default=0.7,
                        type=float,
                        help="threshold of detection model")

        self.ap.add_argument("-m", "--performance_type",
                        default="PERFORMANCE",
                        choices=["PERFORMANCE", "BALANCE", "LOW"],
                        type=str,
                        help="Type of performance mode")

        self.ap.add_argument("-a", "--alignment_flag",
                        default=True,
                        type=bool,
                        help="Alignment flag")

        self.args = vars(self.ap.parse_args())


        self.performance_type = self.args["performance_type"]
        self.alignment_flag = self.args["alignment_flag"]
        self.device = self.args["device"]
        self.detection_threshold = self.args["detection_threshold"]
        self.recognition_threshold = self.args["recognition_threshold"]
        self.model_path = self.args["model"]
        self.app_input = self.args["app_input"]

        if self.device == "gpu":
            self.provider = "CUDAExecutionProvider"
        else:
            self.provider = "CPUExecutionProvider"

        self.detection_model_path = f"{current_dir}\\SAVED_MODELS\\DETECTION_{self.performance_type}.onnx"
        self.face_embedding_model_path = f"{current_dir}\\SAVED_MODELS\\RECOGNITION_{self.performance_type}.onnx"
        self.recognition_model_path = f"{current_dir}\\SAVED_MODELS\\{self.model_path}.onnx"
        self.alignment_model_path = f"{current_dir}\\SAVED_MODELS\\FACE_LANDMARK.onnx"


        ## Load ONNX Models
        self.sess_detection = ort.InferenceSession(self.detection_model_path, providers=[self.provider])
        self.sess_recognition = ort.InferenceSession(self.recognition_model_path, providers=[self.provider])
        self.sess_alignment = ort.InferenceSession(self.alignment_model_path, providers=[self.provider])
        self.sess_face_embedding = ort.InferenceSession(self.face_embedding_model_path, providers=[self.provider])

        self.detection_input_shape = tuple(self.sess_detection.get_inputs()[0].shape[-2:])
        self.alignment_input_shape = tuple(self.sess_alignment.get_inputs()[0].shape[-2:])
        self.recognition_input_shape = tuple(self.sess_recognition.get_inputs()[0].shape[-2:])
        self.face_embed_input_shape = tuple(self.sess_face_embedding.get_inputs()[0].shape[-2:])



        ## Load Emotion Classes
        with open(f'{current_dir}\\SAVED_MODELS\\{self.model_path}.classes', 'r') as f:
            self.EmotionClasses = f.read().strip().split("\n")
        self.FACE_ID_DICT = {}
        self.face_tracker = FaceTracker(self.EmotionClasses)
        self.START_TIME_APP = time.time()

        self.FPS = []
        self.start_stop = True #for start or stop functoin (control in qml)
        self.video_path = None
        self.is_video_mode = False
    
    @pyqtSlot()
    def stop_app(self):
        self.start_stop = False
        self.emotionDataDictForSaveInDataBase = dict()

    @pyqtSlot(str)
    def set_video_path(self, path):
        """Set video path and switch to video mode"""
        self.video_path = path
        self.is_video_mode = True

    @pyqtSlot()
    def process_frame(self, frame):
        """Process a single frame for emotion detection"""
        if frame is None:
            return

        self.orig_image = frame.copy()
        self.image, self.pad_length, self.scale_h, self.scale_w = preprocess_detection(self.orig_image.copy(), self.detection_input_shape)
        self.confidences, self.boxes = onnx_infer(self.image, self.sess_detection)
        self.boxes, self.labels, self.probs = detection_predict(self.detection_input_shape[1], self.detection_input_shape[0], self.confidences, self.boxes, self.detection_threshold)
        
        if len(self.boxes) == 0:
            return
        
        self.time_stamp = time.time()-self.START_TIME_APP
        for box in self.boxes:
            if self.scale_h > self.scale_w:
                box = box - np.array([self.pad_length,0,self.pad_length,0])
                box = np.round(box * np.array([self.scale_h,self.scale_h,self.scale_h,self.scale_h])).astype(np.int32)
            else:
                box = box - np.array([0,self.pad_length,0,self.pad_length])
                box = np.round(box * np.array([self.scale_w,self.scale_w,self.scale_w,self.scale_w])).astype(np.int32)
            
            box[0] = np.clip(box[0], 0, self.orig_image.shape[1])
            box[1] = np.clip(box[1], 0, self.orig_image.shape[0])
            box[2] = np.clip(box[2], 0, self.orig_image.shape[1])
            box[3] = np.clip(box[3], 0, self.orig_image.shape[0])

            self.face_image = self.orig_image[box[1]:box[3],box[0]:box[2],:]

            if self.alignment_flag:
                self.alignment_input = preprocess_alignment(self.orig_image.copy(), box, self.alignment_input_shape)
                self.landmarks = onnx_infer(self.alignment_input, self.sess_alignment)[0]
                self.angle = postprocess_alignment(self.landmarks, self.alignment_input_shape)
                self.face_image = imutils.rotate_bound(self.face_image, self.angle)

            self.face_image_embed = preprocess_face_embedding(self.face_image.copy(), self.face_embed_input_shape, self.performance_type)
            self.face_embed = onnx_infer(self.face_image_embed, self.sess_face_embedding)[0]

            self.face_image_recognition = preprocess_recognition(self.face_image.copy(), self.recognition_input_shape)
            self.face_emotion_logits = onnx_infer(self.face_image_recognition, self.sess_recognition)[0]
            self.face_emotion = postprocess_recognition(self.face_emotion_logits, self.EmotionClasses)
            
            cv2.rectangle(self.frame, (box[0], box[1]), (box[2], box[3]), color=(255,255,255), thickness=2)

            FACE_ID_DICT = self.face_tracker.update(self.FACE_ID_DICT, self.face_embed, self.face_emotion[2], self.time_stamp)

            self.emotionRsult.emit(str(self.face_emotion[0]), float(self.face_emotion[1]), [float(_) for _ in self.face_emotion[2]])

    @pyqtSlot()
    def start_app(self):
        self.start_stop = True
        if self.is_video_mode and self.video_path:
            # Video mode
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("Error: Could not open video file")
                return

            while cap.isOpened() and self.start_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame = frame
                self.process_frame(frame)
                cv2.imshow("Video Processing", self.frame)

                if cv2.waitKey(1) & 0xFF == 27 or not self.start_stop:  # ESC key
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            # Camera mode
            if self.app_input == "0":
                self.cam = cv2.VideoCapture(0)
            else:
                self.cam = cv2.VideoCapture(self.app_input)
            
            cv2.namedWindow("Image")
            while self.start_stop:
                self.ret, self.frame = self.cam.read()
                if not self.ret:
                    break

                self.process_frame(self.frame)
                cv2.imshow("Image", self.frame)

                self.k = cv2.waitKey(1)
                if self.k % 256 == 27 or not self.start_stop:  # ESC key
                    break
            
            self.cam.release()
            cv2.destroyAllWindows()


class BackEndFaceEmotionRecognation(QObject):
    emotionRsult = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob'])
    processingStatus = pyqtSignal(str, float)  # New signal for processing status (status, progress)
    
    def __init__(self):
        super().__init__()
        self.data_buffer = []  # ذخیره داده‌های لحظه‌ای
        self.worker = FaceEmotionRecognation()  # ایجاد نمونه از کلاس Worker
        self.thread = QThread()  # ایجاد ترد جدید
        self.worker.moveToThread(self.thread)  # انتقال Worker به ترد جدید
        self.thread.started.connect(self.worker.start_app)  # اجرای تابع run در ترد
        self.worker.emotionRsult.connect(self.emotionRsult)  # اتصال سیگنال Worker به Backend
        self.worker.emotionRsult.connect(self.saveToMemory)
        self.db = EmotionDatabase(f"dataBase/faceRecognation/emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        self.video_path = None  # مسیر ویدیو
        self.is_processing_video = False  # وضعیت پردازش ویدیو

                # تایمر ذخیره‌سازی دوره‌ای
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.save_to_database)

    @pyqtSlot(str)
    def setVideoPath(self, path):
        """Set the video path and start processing"""
        self.video_path = path.replace("file:///", "").replace("%20", " ")
        if os.path.exists(self.video_path):
            self.is_processing_video = True
            self.worker.set_video_path(self.video_path)
            self.startWorker()
        else:
            print(f"Video file not found: {self.video_path}")

    @pyqtSlot()
    def processVideo(self):
        """Process the video file"""
        if not self.video_path or not os.path.exists(self.video_path):
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened() and self.is_processing_video:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            self.worker.process_frame(frame)
            
            # Update progress
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            self.processingStatus.emit("Processing video...", progress)

            # Add small delay to match video FPS
            time.sleep(1/fps)

        cap.release()
        self.is_processing_video = False
        self.processingStatus.emit("Video processing completed", 100)

    @pyqtSlot()
    def stopVideoProcessing(self):
        """Stop video processing"""
        self.is_processing_video = False
        self.processingStatus.emit("Video processing stopped", 0)

    @pyqtSlot(str, str, int, str)
    def savePersonInfo(self, name, lastName, age, nationalCode):
        self.person_info = {
            "name": name,
            "lastName": lastName,
            "age": age,
            "nationalCode": nationalCode
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = f"dataBase/faceRecognation/face_emotions_{self.person_info.get('name')}_{self.person_info.get('lastName')}_{self.person_info.get('age')}_{self.person_info.get('nationalCode')}_{timestamp}.db"
        print(db_path)
        self.db = EmotionDatabase(db_path)

    @pyqtSlot(str, float, list)
    def saveToMemory(self, emotion_status, demotion_prob, total_prob):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data_buffer.append({
            "timestamp": timestamp,
            "emotion": emotion_status,
            "prob": demotion_prob,
            "probs": total_prob
        })

    @pyqtSlot()
    def save_to_database(self):
        self.db.save_batch(self.data_buffer)
        self.data_buffer.clear()

    @pyqtSlot()
    def generatePdfReport(self):
        data = self.db.fetch_all()
        report = EmotionReport()
        report.generate_report_face_recognation(data)

    @pyqtSlot()
    def startWorker(self):
        """شروع ترد"""
        if not self.thread.isRunning():
            self.save_timer.start(60000)  # هر 60 ثانیه

            self.thread.start()

    @pyqtSlot()
    def stopWorker(self):
        self.save_timer.stop()  # هر 60 ثانیه
        self.worker.stop_app()
        self.thread.quit()
        self.thread.wait()