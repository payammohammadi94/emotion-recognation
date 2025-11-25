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
from .face_reference_manager import FaceReferenceManager
from pathlib import Path
from database_handler import EmotionDatabase
from pdf_utils import EmotionReport

current_dir = Path(__file__).parent
# construct the argument parser and parse the arguments
class FaceEmotionRecognation(QObject):
    emotionRsult = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob'])
    multipleFacesResult = pyqtSignal(list, arguments=['faces_data'])  # Signal for multiple faces
    
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
        
        # Face Reference Manager for snapshots
        self.face_reference_manager = FaceReferenceManager()
        self.face_reference_manager.clear_all_references()  # Clear old snapshots on startup
        
        # Store emotion data for each face
        self.face_data_by_id = {}  # {face_id: {'emotions': [], 'probs': [], ...}}
        
        # Snapshot auto-capture settings
        self.snapshot_delay_frames = 30  # Capture snapshot after 30 frames
        self.face_frame_count = {}  # Track frame count for each face

        self.FPS = []
        self.start_stop = True #for start or stop functoin (control in qml)
        self.video_path = None
        self.is_video_mode = False
        self.cam = None  # Camera object
    
    @pyqtSlot()
    def stop_app(self):
        self.start_stop = False
        self.emotionDataDictForSaveInDataBase = dict()
        self.reset_video_mode()

    @pyqtSlot(str)
    def set_video_path(self, path):
        """Set video path and switch to video mode"""
        self.reset_video_mode()  # Reset before setting new path
        self.video_path = path
        self.is_video_mode = True
    
    @pyqtSlot()
    def reset_video_mode(self):
        """Reset video mode and release capture objects"""
        self.is_video_mode = False
        self.video_path = None
        if self.cam is not None:
            self.cam.release()
            self.cam = None

    def _save_snapshot_for_face(self, face_id, face_image, face_embedding):
        """ذخیره snapshot برای یک چهره"""
        if not self.face_reference_manager.has_reference(face_id):
            snapshot_path = self.face_reference_manager.save_reference_image(face_id, face_image, face_embedding)
            print(f"[✓] Snapshot saved for Face ID {face_id}: {snapshot_path}")
    
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
            # Emit empty list if no faces detected
            self.multipleFacesResult.emit([])
            return
        
        self.time_stamp = time.time()-self.START_TIME_APP
        faces_data = []  # List to store all faces data
        
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
            
            # Compare with reference images first
            matched_id, distance = self.face_reference_manager.compare_with_references(self.face_embed)
            
            if matched_id is not None:
                # Use matched ID from reference
                face_id = matched_id
                self.FACE_ID_DICT, _ = self.face_tracker.update(self.FACE_ID_DICT, self.face_embed, self.face_emotion[2], self.time_stamp)
            else:
                # Use tracker to assign ID
                self.FACE_ID_DICT, face_id = self.face_tracker.update(self.FACE_ID_DICT, self.face_embed, self.face_emotion[2], self.time_stamp)
            
            # Track frame count for snapshot auto-capture
            if face_id not in self.face_frame_count:
                self.face_frame_count[face_id] = 0
            self.face_frame_count[face_id] += 1
            
            # Auto-capture snapshot after delay
            if self.face_frame_count[face_id] == self.snapshot_delay_frames:
                self._save_snapshot_for_face(face_id, self.face_image.copy(), self.face_embed.copy())
            
            # Store emotion data for this face
            if face_id not in self.face_data_by_id:
                self.face_data_by_id[face_id] = {
                    'emotions': [],
                    'probs': [],
                    'emotion_probs': []
                }
            
            self.face_data_by_id[face_id]['emotions'].append(self.face_emotion[0])
            self.face_data_by_id[face_id]['probs'].append(self.face_emotion[1])
            self.face_data_by_id[face_id]['emotion_probs'].append(self.face_emotion[2])
            
            # Draw bounding box
            cv2.rectangle(self.frame, (box[0], box[1]), (box[2], box[3]), color=(255,255,255), thickness=2)
            
            # Draw emotion and face ID on image
            emotion_text = f"ID:{face_id} {self.face_emotion[0]}"
            prob_text = f"{self.face_emotion[1]:.2f}"
            
            # Calculate text position
            text_y = box[1] - 10 if box[1] > 30 else box[3] + 25
            
            # Draw emotion text
            cv2.putText(self.frame, emotion_text, (box[0], text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(self.frame, prob_text, (box[0], text_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add to faces_data list
            faces_data.append({
                'face_id': face_id,
                'emotion': self.face_emotion[0],
                'emotion_prob': self.face_emotion[1],
                'emotion_probs': [float(_) for _ in self.face_emotion[2]],
                'box': box.tolist()
            })
            
            # Emit individual emotion result (for backward compatibility)
            self.emotionRsult.emit(str(self.face_emotion[0]), float(self.face_emotion[1]), [float(_) for _ in self.face_emotion[2]])
        
        # Emit multiple faces result
        self.multipleFacesResult.emit(faces_data)

    @pyqtSlot()
    def start_app(self):
        self.start_stop = True
        if self.is_video_mode and self.video_path:
            # Video mode
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("Error: Could not open video file")
                self.reset_video_mode()
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
            self.reset_video_mode()  # Reset after video ends
        else:
            # Camera mode - release any existing camera first
            if self.cam is not None:
                self.cam.release()
            
            if self.app_input == "0":
                self.cam = cv2.VideoCapture(0)
            else:
                self.cam = cv2.VideoCapture(self.app_input)
            
            if not self.cam.isOpened():
                print("Error: Could not open camera")
                return
            
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
            
            if self.cam is not None:
                self.cam.release()
            cv2.destroyAllWindows()


class BackEndFaceEmotionRecognation(QObject):
    emotionRsult = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob'])
    multipleFacesResult = pyqtSignal(list, arguments=['faces_data'])  # Signal for multiple faces
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
        self.worker.multipleFacesResult.connect(self.multipleFacesResult)  # Connect multiple faces signal
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
        # تبدیل numpy array به لیست برای JSON serialization
        probs = total_prob
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        elif not isinstance(probs, list):
            probs = list(probs)
        
        self.data_buffer.append({
            "timestamp": timestamp,
            "emotion": emotion_status,
            "prob": float(demotion_prob),
            "probs": probs
        })

    @pyqtSlot()
    def save_to_database(self):
        """ذخیره داده‌ها در دیتابیس (فقط data_buffer)"""
        if self.data_buffer:
            try:
                self.db.save_batch(self.data_buffer)
                self.data_buffer.clear()
            except Exception as e:
                print(f"[-] Error saving to database: {e}")
    
    def save_face_data_to_database(self):
        """ذخیره داده‌های هر صورت در دیتابیس"""
        if hasattr(self.worker, 'face_data_by_id'):
            face_data_buffer = []  # استفاده از buffer جداگانه برای جلوگیری از recursion
            for face_id, face_data in self.worker.face_data_by_id.items():
                emotions_count = len(face_data.get('emotions', []))
                print(f"[*] Face ID {face_id}: {emotions_count} emotions to save")
                for i, emotion in enumerate(face_data.get('emotions', [])):
                    # تبدیل numpy array به لیست برای JSON serialization
                    probs = face_data.get('emotion_probs', [])[i] if i < len(face_data.get('emotion_probs', [])) else []
                    if isinstance(probs, np.ndarray):
                        probs = probs.tolist()
                    elif not isinstance(probs, list):
                        probs = list(probs)
                    
                    face_data_buffer.append({
                        'face_id': int(face_id),  # اطمینان از int بودن
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'emotion': emotion,
                        'prob': float(face_data.get('probs', [])[i]) if i < len(face_data.get('probs', [])) else 0.0,
                        'probs': probs
                    })
            if face_data_buffer:
                try:
                    print(f"[*] Saving {len(face_data_buffer)} face data items to database...")
                    self.db.save_batch(face_data_buffer)
                    print(f"[✓] Successfully saved {len(face_data_buffer)} face data items")
                except Exception as e:
                    print(f"[-] Error saving face data to database: {e}")
            else:
                print("[*] No face data to save")
        else:
            print("[*] No face_data_by_id attribute in worker")

    @pyqtSlot()
    def generatePdfReport(self):
        """تولید گزارش PDF برای تمام صورت‌ها"""
        # ذخیره داده‌های باقی‌مانده
        print("[*] Saving remaining face data to database...")
        self.save_face_data_to_database()
        
        # ذخیره data_buffer هم
        if self.data_buffer:
            print(f"[*] Saving {len(self.data_buffer)} items from data_buffer...")
            self.save_to_database()
        
        # بررسی وجود snapshot ها
        if hasattr(self.worker, 'face_reference_manager'):
            snapshot_ids = self.worker.face_reference_manager.get_all_reference_ids()
            print(f"[*] Found {len(snapshot_ids)} snapshots: {snapshot_ids}")
            if snapshot_ids:
                # تولید گزارش با snapshot ها
                self.generatePdfReportsForAllFaces()
            else:
                # تولید گزارش عادی
                data = self.db.fetch_all()
                print(f"[*] No snapshots, fetching all data: {len(data)} items")
                if data:
                    report = EmotionReport()
                    report.generate_report_face_recognation(data)
                else:
                    print("[-] No data to generate report.")
        else:
            # تولید گزارش عادی اگر face_reference_manager وجود ندارد
            data = self.db.fetch_all()
            print(f"[*] No face_reference_manager, fetching all data: {len(data)} items")
            if data:
                report = EmotionReport()
                report.generate_report_face_recognation(data)
            else:
                print("[-] No data to generate report.")
    
    @pyqtSlot(int)
    def generatePdfReportForFace(self, face_id):
        """تولید گزارش PDF برای یک صورت خاص"""
        if not hasattr(self.worker, 'face_data_by_id') or face_id not in self.worker.face_data_by_id:
            print(f"[-] No data found for face ID {face_id}")
            return
        
        # دریافت داده‌های صورت از دیتابیس
        data = self.db.fetch_by_face_id(face_id)
        if not data:
            print(f"[-] No database data found for face ID {face_id}")
            return
        
        # دریافت snapshot
        snapshot_path = None
        if hasattr(self.worker, 'face_reference_manager'):
            ref_image = self.worker.face_reference_manager.load_reference_image(face_id)
            if ref_image is not None:
                snapshot_path = self.worker.face_reference_manager.reference_images.get(face_id)
        
        # تولید گزارش
        report = EmotionReport()
        report.generate_report_for_single_face(face_id, data, snapshot_path)
    
    @pyqtSlot()
    def generatePdfReportsForAllFaces(self):
        """تولید گزارش PDF برای تمام صورت‌ها با snapshot ها"""
        if not hasattr(self.worker, 'face_reference_manager'):
            print("[-] Face reference manager not available")
            return
        
        snapshot_ids = self.worker.face_reference_manager.get_all_reference_ids()
        if not snapshot_ids:
            print("[-] No snapshots found")
            return
        
        # ذخیره داده‌های باقی‌مانده قبل از تولید گزارش
        self.save_face_data_to_database()
        
        # جمع‌آوری داده‌ها و snapshot ها
        faces_data = []
        for face_id in snapshot_ids:
            # تبدیل face_id به int اگر string است
            try:
                face_id_int = int(face_id)
            except (ValueError, TypeError):
                face_id_int = face_id
            
            data = self.db.fetch_by_face_id(face_id_int)
            snapshot_path = self.worker.face_reference_manager.reference_images.get(face_id)
            
            print(f"[*] Face ID {face_id_int}: {len(data) if data else 0} data points, snapshot: {snapshot_path is not None}")
            
            # اگر داده‌ای در دیتابیس نیست، از face_data_by_id استفاده کن
            if not data and hasattr(self.worker, 'face_data_by_id') and face_id_int in self.worker.face_data_by_id:
                face_data = self.worker.face_data_by_id[face_id_int]
                # تبدیل به فرمت دیتابیس
                data = []
                for i, emotion in enumerate(face_data.get('emotions', [])):
                    probs = face_data.get('emotion_probs', [])[i] if i < len(face_data.get('emotion_probs', [])) else []
                    if isinstance(probs, np.ndarray):
                        probs = probs.tolist()
                    data.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'emotion': emotion,
                        'prob': float(face_data.get('probs', [])[i]) if i < len(face_data.get('probs', [])) else 0.0,
                        'probs': probs,
                        'face_id': face_id_int
                    })
                print(f"[*] Using face_data_by_id for Face ID {face_id_int}: {len(data)} data points")
            
            if snapshot_path:
                faces_data.append({
                    'face_id': face_id_int,
                    'data': data if data else [],
                    'snapshot_path': snapshot_path
                })
        
        if faces_data:
            report = EmotionReport()
            report.generate_report_with_snapshots(faces_data)
            
            # حذف snapshot ها بعد از تولید گزارش
            self.worker.face_reference_manager.clear_all_references()
        else:
            print("[-] No valid data found for report generation")
            print(f"[*] Snapshot IDs: {snapshot_ids}")
            print(f"[*] Total faces_data collected: {len(faces_data)}")
    
    @pyqtSlot()
    def capture_snapshot(self, face_id):
        """دستی capture snapshot برای یک صورت"""
        if hasattr(self.worker, 'face_reference_manager'):
            # این متد باید از worker فراخوانی شود
            pass
    
    @pyqtSlot(int, result='QVariant')
    def getFaceData(self, face_id):
        """دریافت داده‌های یک صورت"""
        if hasattr(self.worker, 'face_data_by_id') and face_id in self.worker.face_data_by_id:
            return self.worker.face_data_by_id[face_id]
        return None
    
    @pyqtSlot(result='QVariantList')
    def getAllFaceIds(self):
        """دریافت لیست تمام face ID ها"""
        if hasattr(self.worker, 'face_data_by_id'):
            return list(self.worker.face_data_by_id.keys())
        return []
    
    @pyqtSlot(bool)
    def setAutoCapture(self, enabled):
        """فعال/غیرفعال کردن auto-capture"""
        if hasattr(self.worker, 'snapshot_delay_frames'):
            self.worker.snapshot_delay_frames = 30 if enabled else float('inf')
    
    @pyqtSlot()
    def resetToWebcamMode(self):
        """Reset به حالت webcam"""
        if hasattr(self.worker, 'reset_video_mode'):
            self.worker.reset_video_mode()

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
        if hasattr(self.worker, 'reset_video_mode'):
            self.worker.reset_video_mode()
        self.thread.quit()
        self.thread.wait()