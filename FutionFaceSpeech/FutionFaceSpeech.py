from PyQt6.QtCore import QThread,QObject,pyqtSignal,pyqtSlot
from FaceRecognation.run_app import FaceEmotionRecognation
from SpeechEmotionRecognation.SpeechEmotionRecognation import SpeechEmotionRecognation

import numpy as np



class FusionManager(QObject):
    finalResult = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob'])  # ارسال نتیجه نهایی به QML

    def __init__(self):
        super().__init__()
        self.image_probs = None
        self.audio_probs = None
        self.alpha = 0.6  # مقدار پیش‌فرض alpha
        self.beta = 0.4   # مقدار پیش‌فرض beta

    @pyqtSlot(float, float)
    def setWeights(self, alpha, beta):
        """تغییر مقادیر وزن‌های alpha و beta"""
        if 0 <= alpha <= 1 and 0 <= beta <= 1:
            self.alpha = alpha
            self.beta = beta

    @pyqtSlot(str,float, list)
    def updateImageResult(self, emotionPredict, prob_value, probs):
        self.image_probs = np.array(probs)
        self.fuseResults()

    @pyqtSlot(str,float, list)
    def updateAudioResult(self, emotionPredict,prob_value,probs):
        probs.pop(-1)
        probs.pop(-4)
        self.audio_probs = np.array(probs)
        self.fuseResults()

    def fuseResults(self):
        if self.image_probs is not None and self.audio_probs is not None and len(self.image_probs) == len(self.audio_probs):
            # ترکیب نتایج با وزن‌های قابل تنظیم
            final_probs = (self.alpha * self.image_probs) + (self.beta * self.audio_probs)

            final_class = np.argmax(final_probs)
            emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
            emotion_label = emotions[final_class]

            self.finalResult.emit(emotion_label, float(final_probs[final_class]), [float(_) for _ in final_probs])

            # پاک کردن نتایج بعد از پردازش
            self.image_probs = None
            self.audio_probs = None




class BackEndFution(QObject):
    finalResult = pyqtSignal(str,float, list, arguments=['emotion_status','demotion_prob','total_prob']) 
    # ایجاد ترد برای پردازش تصویر
    def __init__(self):
        super().__init__()
        
        self.face_recognizer = FaceEmotionRecognation()
        self.face_thread = QThread()
        self.face_recognizer.moveToThread(self.face_thread)
        self.face_thread.started.connect(self.face_recognizer.start_app)


            # ایجاد ترد برای پردازش صوت
        self.audio_recognizer = SpeechEmotionRecognation()
        self.audio_thread = QThread()
        self.audio_recognizer.moveToThread(self.audio_thread)
        self.audio_thread.started.connect(self.audio_recognizer.run)

    # ایجاد مدیریت فیوژن
        self.fusion_manager = FusionManager()
        self.face_recognizer.emotionRsult.connect(self.fusion_manager.updateImageResult)
        self.audio_recognizer.emotionRsultSpeech.connect(self.fusion_manager.updateAudioResult)
        self.fusion_manager.finalResult.connect(self.finalResult)
    @pyqtSlot()
    def startWorker(self):
        self.audio_thread.start()
        self.face_thread.start()


    @pyqtSlot()
    def stopWorker(self):

        self.face_recognizer.stop_app()
        self.face_thread.quit()
        self.face_thread.wait()

        self.audio_recognizer.stop()
        self.audio_thread.quit()
        self.audio_thread.wait()

    @pyqtSlot(float, float)
    def setWeights(self, alpha, beta):
        """تغییر مقادیر وزن‌ها در FusionManager"""
        self.fusion_manager.setWeights(alpha, beta)