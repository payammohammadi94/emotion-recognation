
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf
from funasr import AutoModel

from PyQt6.QtCore import QUrl,QTimer,QObject,pyqtSignal,pyqtSlot,QThread
from database_handler import EmotionDatabase
from pdf_utils import EmotionReport

current_dir = Path(__file__).parent




class SpeechEmotionRecognation(QObject):
    emotionRsultSpeech = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob'])
    def __init__(self):
        super().__init__()

    # Emotion mapping for translation
        self.emotion_map = {
            "angry": "Angry",
            "disgusted": "Disgusted",
            "fearful": "Fearful",
            "happy": "Happy",
            "neutral": "Neutral",
            "other": "Other",
            "sad": "Sad",
            "surprised": "Surprised",
            "unknown": "Unknown"
        }

    # Pre-trained model
        self.model = AutoModel(model=f"{current_dir}\\emotion2vec_plus_large")
        
        self.sr = 16000  # Sampling rate
        # Audio data for real-time mode
        self.audio_buffer = np.array([])  # Audio data buffer
        self.realtime_mode = False  # Real-time mode status
        self.start_stop = True #for start or stop functoin (control in qml)
        self.audio_file_path = None  # مسیر فایل صوتی
        self.is_file_mode = False  # حالت فایل یا real-time

    def extract_emotion(self,lbl):
        self.parts = lbl.split("/")
        if len(self.parts) > 1:
            return self.parts[1]
        else:
            return self.parts[0]

    def app(self,audio_buffer):
        # If there is not enough data for processing, do nothing
        if audio_buffer.size < self.sr:
            return

        # Only check the last 1 second of data
        self.segment = self.audio_buffer[-self.sr:]
        
        # Run the model inference
        self.res = self.model.inference(self.segment, disable_pbar=True)
        self.labels = self.res[0]["labels"]
        self.scores = self.res[0]["scores"]
        self.probs = np.array(self.scores)
        
        self.emos = {}
        for lbl, p in zip(self.labels, self.probs):
            self.emos[self.extract_emotion(lbl)] = p
        
        # Normalize values for each emotion
        self.sum_probs = sum(self.emos.values())
        if self.sum_probs > 0:
            self.normalized_vals = {k: (v / self.sum_probs) * 100 for k, v in self.emos.items()}
        else:
            self.normalized_vals = {k: 0 for k in self.emos.keys()}

        # Print the predicted results in console
        # print("Real-time mode active: Emotions detected")
        
        # Sort emotions by percentage, print primary emotion first
        self.sorted_emos = sorted(self.normalized_vals.items(), key=lambda x: x[1], reverse=True)
        
        self.primary_emotion, self.primary_value = self.sorted_emos[0]
        # print(f"Primary Emotion: {self.primary_emotion} with {int(round(self.primary_value))}%")
        
        self.emotionRsultSpeech.emit(str(self.primary_emotion),float(self.primary_value)/100,[float(_) for _ in self.probs])

        
        # Print other emotions with their respective percentages
        # for emotion, value in self.sorted_emos[1:]:
        #     print(f"{int(round(value))}% {emotion}")

    # Function to record audio data
    def append_audio_data(self,data_chunk):
        
        if self.audio_buffer.size == 0:
            self.audio_buffer = data_chunk
        else:
            self.audio_buffer = np.concatenate((self.audio_buffer, data_chunk))

        self.max_len = self.sr  # Maximum audio data length
        if len(self.audio_buffer) > self.max_len:
            self.audio_buffer = self.audio_buffer[-self.max_len:]

    # Function to start recording and call the app function
    @pyqtSlot()
    def stop(self):
        self.start_stop = False

    @pyqtSlot(str)
    def set_audio_path(self, path):
        """تنظیم مسیر فایل صوتی و تغییر به حالت فایل"""
        self.audio_file_path = path
        self.is_file_mode = True
        print(f"[*] Audio file path set: {path}")

    @pyqtSlot()
    def reset_audio_mode(self):
        """Reset به حالت real-time"""
        self.is_file_mode = False
        self.audio_file_path = None
        print("[*] Reset to real-time mode")

    def process_audio_file(self, file_path):
        """پخش فایل صوتی و ضبط صدای خروجی برای تحلیل"""
        try:
            print(f"[*] Playing audio file: {file_path}")
            
            # بارگذاری فایل صوتی با soundfile
            print("[*] Loading audio file...")
            try:
                audio_data, original_sr = sf.read(file_path)
                print(f"[*] Audio loaded - Duration: {len(audio_data)/original_sr:.2f}s, Sample rate: {original_sr}Hz")
            except Exception as e:
                print(f"[-] Error loading audio file: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # تبدیل به mono اگر استریو باشد
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                print("[*] Converted stereo to mono")
            
            # تبدیل به sampling rate مورد نیاز برای پخش
            playback_sr = original_sr
            if original_sr != self.sr:
                print(f"[*] Resampling from {original_sr}Hz to {self.sr}Hz for playback...")
                try:
                    from scipy import signal
                    num_samples = int(len(audio_data) * self.sr / original_sr)
                    audio_data = signal.resample(audio_data, num_samples)
                    playback_sr = self.sr
                    print(f"[✓] Resampling completed - New duration: {len(audio_data)/self.sr:.2f}s")
                except ImportError:
                    # استفاده از numpy interpolation
                    print("[*] scipy not available, using numpy interpolation...")
                    old_indices = np.linspace(0, len(audio_data) - 1, len(audio_data))
                    new_length = int(len(audio_data) * self.sr / original_sr)
                    new_indices = np.linspace(0, len(audio_data) - 1, new_length)
                    audio_data = np.interp(new_indices, old_indices, audio_data)
                    playback_sr = self.sr
                    print(f"[✓] Resampling completed - New duration: {len(audio_data)/self.sr:.2f}s")
            
            # پاک کردن buffer قبلی
            self.audio_buffer = np.array([])
            
            # پخش فایل و همزمان ضبط از میکروفن
            print("[*] Playing audio and recording from microphone...")
            print("[*] Please ensure your speakers are on and microphone can hear the playback")
            
            duration = len(audio_data) / playback_sr
            print(f"[*] Audio duration: {duration:.2f} seconds")
            
            # شروع ضبط همزمان با پردازش real-time
            frame_count = 0
            def record_callback(indata, frames, time, status):
                nonlocal frame_count
                if status:
                    print(f"[!] Recording status: {status}")
                # اضافه کردن داده به buffer
                self.append_audio_data(indata[:, 0])
                frame_count += frames
                # پردازش real-time (هر 1 ثانیه)
                if self.audio_buffer.size >= self.sr:
                    try:
                        self.app(self.audio_buffer)
                    except Exception as e:
                        print(f"[-] Error in app processing: {e}")
            
            # شروع ضبط
            with sd.InputStream(samplerate=self.sr, channels=1, callback=record_callback):
                # پخش فایل
                print("[*] Starting playback...")
                sd.play(audio_data, samplerate=playback_sr)
                
                # صبر تا پخش تمام شود و همزمان پردازش انجام شود
                duration_seconds = len(audio_data) / playback_sr
                import time
                elapsed = 0
                while elapsed < duration_seconds + 0.5 and self.start_stop:
                    time.sleep(0.1)
                    elapsed += 0.1
                
                # اطمینان از توقف پخش
                sd.wait()
                time.sleep(0.5)  # صبر کمی بیشتر برای ضبط کامل
            
            print("[✓] Audio file playback and analysis completed")
            
        except Exception as e:
            print(f"[-] Error processing audio file: {e}")
            import traceback
            traceback.print_exc()

    @pyqtSlot()
    def run(self):
        self.start_stop = True
        
        print(f"[*] run() called - is_file_mode: {self.is_file_mode}, audio_file_path: {self.audio_file_path}")
        
        if self.is_file_mode and self.audio_file_path:
            # حالت فایل صوتی
            print("[*] Starting audio file processing...")
            try:
                self.process_audio_file(self.audio_file_path)
                print("[✓] Audio file processing completed")
            except Exception as e:
                print(f"[-] Error in process_audio_file: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Reset بعد از پردازش
                self.reset_audio_mode()
                print("[*] Audio mode reset, ready for next operation")
        else:
            # حالت real-time از میکروفن
            print("[*] Starting real-time recording from microphone...")
            try:
                with sd.InputStream(channels=1, samplerate=self.sr, callback=self.callback):
                    while self.start_stop:
                        sd.sleep(200)
                print("[*] Real-time recording stopped")
            except Exception as e:
                print(f"[-] Error in real-time recording: {e}")
                import traceback
                traceback.print_exc()


    def callback(self,indata, frames, time, status):
        self.append_audio_data(indata[:, 0])  # Add recorded data to buffer
        self.app(self.audio_buffer)  # Perform real-time analysis by calling the app function


class BackEnd(QObject):
    dataUpdated = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob'])
    def __init__(self):
        super().__init__()
        self.data_buffer = []  # ذخیره داده‌های لحظه‌ای
        self.worker = SpeechEmotionRecognation()  # ایجاد نمونه از کلاس Worker
        self.thread = QThread()  # ایجاد ترد جدید
        self.worker.moveToThread(self.thread)  # انتقال Worker به ترد جدید
        self.thread.started.connect(self.worker.run)  # اجرای تابع run در ترد
        self.worker.emotionRsultSpeech.connect(self.dataUpdated)  # اتصال سیگنال Worker به Backend
        self.worker.emotionRsultSpeech.connect(self.saveToMemory)
        self.db = EmotionDatabase(f"dataBase/voiceRecognation/voice_emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")

        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.save_to_database)
        self.save_timer.start(60000)  # هر 60 ثانیه

    @pyqtSlot(str, str, int, str)
    def savePersonInfo(self, name, lastName, age, nationalCode):
        self.person_info = {
            "name": name,
            "lastName": lastName,
            "age": age,
            "nationalCode": nationalCode
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = f"dataBase/voiceRecognation/voice_emotions_{self.person_info.get('name')}_{self.person_info.get('lastName')}_{self.person_info.get('age')}_{self.person_info.get('nationalCode')}_{timestamp}.db"
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
        if self.data_buffer:
            try:
                self.db.save_batch(self.data_buffer)
                self.data_buffer.clear()
            except Exception as e:
                print(f"[-] Error saving to database: {e}")

    @pyqtSlot()
    def generatePdfReport(self):
        data = self.db.fetch_all()
        report = EmotionReport()
        report.generate_report_voice_recognation(data)

    @pyqtSlot(str)
    def setAudioPath(self, path):
        """تنظیم مسیر فایل صوتی و شروع پردازش"""
        import os
        print(f"[*] setAudioPath called with: {path}")
        self.audio_file_path = path.replace("file:///", "").replace("%20", " ")
        print(f"[*] Processed path: {self.audio_file_path}")
        
        if os.path.exists(self.audio_file_path):
            print(f"[✓] Audio file exists: {self.audio_file_path}")
            # اگر thread در حال اجرا است، ابتدا stop کنیم
            if self.thread.isRunning():
                print("[*] Thread is running, stopping it first...")
                self.worker.stop()
                self.thread.quit()
                self.thread.wait()
                print("[✓] Thread stopped")
            
            # تنظیم مسیر فایل در worker
            self.worker.set_audio_path(self.audio_file_path)
            print("[*] Starting worker with audio file...")
            self.startWorker()
        else:
            print(f"[-] Audio file not found: {self.audio_file_path}")

    @pyqtSlot()
    def startWorker(self):
        """شروع ترد"""
        if not self.thread.isRunning():
            self.save_timer.start(10000)  # هر 60 ثانیه
            self.thread.start()

    @pyqtSlot()
    def stopWorker(self):
        """متوقف کردن ترد"""
        self.save_timer.stop()  # هر 60 ثانیه
        self.worker.stop()
        
        # Stop any playing audio
        try:
            sd.stop()
        except:
            pass
        
        # Reset audio mode when stopping
        if hasattr(self.worker, 'reset_audio_mode'):
            self.worker.reset_audio_mode()
        
        # Wait for thread to finish
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
    
    @pyqtSlot()
    def resetToMicrophoneMode(self):
        """Reset به حالت میکروفن"""
        # Stop worker if running
        if self.thread.isRunning():
            self.stopWorker()
        
        # Reset audio mode
        if hasattr(self.worker, 'reset_audio_mode'):
            self.worker.reset_audio_mode()
        
        # Stop any playing audio
        try:
            sd.stop()
        except:
            pass
        
        print("[*] Reset to microphone mode")

