import os
import sys
import json

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtQml import QQmlApplicationEngine

from FaceRecognation.run_app import BackEndFaceEmotionRecognation
from SpeechEmotionRecognation.SpeechEmotionRecognation import BackEnd
from FutionFaceSpeech.FutionFaceSpeech import BackEndFution
from eegEmotionRecognation.fineeg import FileHandler
import uvicorn
from api import app

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    faceEmotionRecognation = BackEndFaceEmotionRecognation()
    backEnd = BackEnd()
    backEndFution = BackEndFution()
    fileHandler = FileHandler()
    engine.load(QUrl.fromLocalFile(os.path.join(CURRENT_DIR, "main.qml")))
    # app.setWindowIcon(QIcon(os.path.join(CURRENT_DIR, "appicon.png")))
    engine.rootObjects()[0].setProperty("backEnd",backEnd)
    engine.rootObjects()[0].setProperty("faceEmotionRecognation",faceEmotionRecognation)
    engine.rootObjects()[0].setProperty("backEndFution",backEndFution)
    engine.rootObjects()[0].setProperty("fileHandler",fileHandler)

    
    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())