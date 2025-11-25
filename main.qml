import QtQuick 2.15
import QtQuick.Controls 2.15
import QtCharts 2.15
import QtQuick.Dialogs
import QtQuick.Controls.Material 2.15

ApplicationWindow {
    property color slideColor : "#092145"
    property color titleText : "#FBF5DD"
    property color subTitle : "#CBA35C"
    property bool deviceStatus : false
    
    // property QtObject eegProvider : eegProvider
    property QtObject faceEmotionRecognation : faceEmotionRecognation
    property QtObject backEnd : backEnd
    property QtObject backEndFution : backEndFution
    property QtObject fileHandler : fileHandler
    
    
    id:mainWindow
    visible: true
    width: 1300
    height: 800
    title: "EmotionRecognation 1.1V"
    
    
    
    Image {
        anchors.fill: parent
        source: "./images/background.jpg" // مسیر عکس
        fillMode: Image.PreserveAspectCrop
    }
    
    
    StackView {
        id: stackView
        anchors.fill: parent
        initialItem: firstPage
    }
    
    //first page start page
    Component {
        id: firstPage
        Rectangle {
            width: parent.width
            height: parent.height
            color: "transparent" // شفاف برای نشان دادن بک‌گراند
            
            Text {
                id: projectName
                text: qsTr("Emotion Recognition")
                font.family: "Arial"
                font.pointSize: 24
                font.bold:true
                color: titleText
                anchors{
                    top: parent.top
                    topMargin:parent.height/2
                    left:parent.left
                    leftMargin: 35
                    
                }
            }
            Text {
                id: centerName
                text: qsTr("Electronic warfare & Cybernetics Research Center   ")
                font.family: "Arial"
                font.pointSize: 20
                font.bold:true
                color: subTitle
                anchors{
                    top: projectName.bottom
                    left:projectName.left
                    topMargin: 10
                }
            }
            Button {
                id:startAppButtonId
                anchors{
                    top: centerName.bottom
                    topMargin: 10
                    left: centerName.left
                    leftMargin: 0
                }
                width: 120
                height: 40
                background: Rectangle{
                    color: titleText
                    radius: 10
                }
                // تنظیم رنگ متن
                contentItem: Text {
                    text: qsTr("Start app")
                    color: "black" // رنگ متن
                    font.pixelSize: parent.font.pixelSize
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                onClicked: stackView.push(secondPage)
            }
            
        }
    }
    //second page category page
    Component {
        id: secondPage
        Rectangle {
            width: parent.width
            height: parent.height
            color: "transparent"
            
            Rectangle{
                id:slide1Id
                width: 250
                height: 250
                color: slideColor
                radius: 15
                anchors{
                    top:parent.top
                    topMargin: parent.height/4
                    left: parent.left
                    leftMargin: 100
                }
                Image {
                    id: backgroundSlide1
                    source: "./images/face_recognation.png"
                    fillMode: Image.PreserveAspectCrop
                    anchors.fill: parent
                }
                MouseArea{
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked:   stackView.push(threePage)
                }
            }
            
            Rectangle{
                id:slide2Id
                width: 250
                height: 250
                color: slideColor
                radius: 15
                anchors{
                    top:slide1Id.bottom
                    topMargin: 20
                    left: slide1Id.left
                }
                Image {
                    id: backgroundSlide2
                    source: "./images/eeg_recognation.jpeg"
                    fillMode: Image.PreserveAspectCrop
                    anchors.fill: parent
                }
                MouseArea{
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked:   stackView.push(sixPage)
                }
            }
            
            Rectangle{
                id:slide3Id
                width: 250
                height: 250
                color: slideColor
                radius: 15
                anchors{
                    top:parent.top
                    topMargin: parent.height/4
                    left: slide1Id.right
                    leftMargin: 20
                }
                Image {
                    id: backgroundSlide3
                    source: "./images/voice_recognation.jpg"
                    fillMode: Image.PreserveAspectCrop
                    anchors.fill: parent
                }
                MouseArea{
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked:   stackView.push(fourPage)
                }
            }
            
            Rectangle{
                id:slide4Id
                width: 250
                height: 250
                color: slideColor
                radius: 15
                anchors{
                    top:slide3Id.bottom
                    topMargin: 20
                    left: slide2Id.right
                    leftMargin: 20
                }
                Image {
                    id: backgroundSlide4
                    source: "./images/fution.jpg"
                    fillMode: Image.PreserveAspectCrop
                    anchors.fill: parent
                }
                MouseArea{
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked:   stackView.push(fivePage)
                }
            }
        }
    }

    //three page face
    Component {
        id: threePage
        Rectangle {
            QtObject {
                id: emotionDataThree
                property var emotions: ({
                                            "anger": 0,
                                            "disgust": 0,
                                            "fear": 0,
                                            "happiness": 0,
                                            "neutral": 0,
                                            "sadness": 0,
                                            "surprise": 0,
                                        });
                signal emotionsQmlChanged();
            }
            property var newEmotions : {
                "anger": 0,
                "disgust": 0,
                "fear": 0,
                "happiness": 0,
                "neutral": 0,
                "sadness": 0,
                "surprise": 0,
            };
            
            property var emotionKeys: ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
            
            id:page3Id
            property bool startStop : false
            property double percentageEmotion: 0;
            property string emotionStatus: "Empty";
            anchors.fill: parent
            color: "#212121"
            Text {
                id: centerName
                text: qsTr("Real Time Face Emotion Detection")
                font.family: "Arial"
                font.pointSize: 18
                font.bold:true
                color: titleText
                anchors{
                    top: page3Id.top
                    topMargin: 20
                    horizontalCenter: parent.horizontalCenter
                }
            }
            
            Rectangle{
                
                id:circulePlot
                width: parent.width / 4 + 20
                height: parent.height / 1.8
                color: "#424242"
                
                border.color: "#424242"
                border.width: 1.5
                
                anchors{
                    top: centerName.bottom
                    topMargin: 30
                    left: parent.left
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                
                Column {
                    width : parent.width - 10
                    height : parent.height -10
                    spacing: 1
                    Grid {
                        id: gridThree
                        columns: 2      // دو ستون
                        spacing: 2
                        //anchors.horizontalCenter: parent.horizontalCenter
                        Repeater {
                            model: emotionKeys
                            delegate: Item {
                                width: 180
                                height: 100
                                // تعیین رنگ هر احساس بر اساس کلید آن
                                property string emotionColor: {
                                    switch(modelData) {
                                    case "anger": return "red";
                                    case "disgust": return "yellow";
                                    case "fear": return "purple";
                                    case "happiness": return "green";
                                    case "neutral": return "orange";
                                    case "sadness": return "blue";
                                    case "surprise": return "pink";
                                    default: return "cyan";
                                    }
                                }
                                
                                Canvas {
                                    id:progressCanvasThree
                                    anchors.fill: parent
                                    
                                    Connections {
                                        target: emotionDataThree
                                        onEmotionsQmlChanged: progressCanvasThree.requestPaint()
                                    }
                                    onPaint: {
                                        var ctx = getContext("2d");
                                        ctx.clearRect(0, 0, width, height);
                                        
                                        var percentage = emotionDataThree.emotions[modelData] || 0;  // درصد مربوط به احساس فعلی
                                        var startAngle = Math.PI;                // شروع از ۱۸۰ درجه (رادیان)
                                        var endAngle = Math.PI * (1 + (percentage / 100)); // انتهای قوس بر اساس درصد
                                        var centerX = width / 2;
                                        var centerY = height - 10;
                                        var radius = 50;
                                        
                                        // رسم نیم‌دایره پس‌زمینه (رنگ خاکستری)
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = "gray";
                                        ctx.stroke();
                                        
                                        // رسم نیم‌دایره پیشرفت (با رنگ اختصاصی)
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = emotionColor;
                                        ctx.stroke();
                                    }
                                }
                                
                                Text {
                                    id :textCircule;
                                    text: modelData + ": ";
                                    color: "white"
                                    anchors.horizontalCenter: parent.horizontalCenter;
                                    anchors.top: progressCanvasThree.bottom;
                                    font.bold: true;
                                    Connections {
                                        target: emotionDataThree
                                        onEmotionsQmlChanged: textCircule.text = modelData + ": " + Math.round(emotionDataThree.emotions[modelData] || 0) + "%"
                                    }
                                }
                            }
                        }
                    }
                }
                
            }
            
            Rectangle {
                id: linePlot
                width: parent.width / 1.5 + 20
                height: parent.height / 1.8
                color: "#424242"
                border.color: "#424242"
                border.width: 1.5
                
                anchors {
                    top: centerName.bottom
                    topMargin: 30
                    left: circulePlot.right
                    leftMargin: 20
                }
                
                ChartView {
                    id: chartView
                    anchors.fill: parent
                    antialiasing: true
                    title: "Emotion Probabilities Over Time"
                    backgroundColor: "#424242"
                    legend.visible: true
                    legend.labelColor: "white"
                    titleColor: "white"
                    
                    property int maxPoints: 100
                    property real startTime: 0
                    property int timeCounter: 0  // زمان مستقل برای محور x
                    
                    ValueAxis {
                        id: xAxis
                        min: chartView.startTime
                        max: chartView.startTime + chartView.maxPoints
                        tickCount: 11
                        titleText: "Time"
                        labelsColor: "white"
                    }
                    
                    ValueAxis {
                        id: yAxis
                        min: 0
                        max: 100
                        tickCount: 11
                        titleText: "Probability (%)"
                        labelsColor: "white"
                    }
                    
                    // سری‌ها
                    LineSeries { id: seriesAnger; name: "Anger"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesDisgust; name: "Disgust"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesFear; name: "Fear"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesHappiness; name: "Happiness"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesNeutral; name: "Neutral"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSadness; name: "Sadness"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSurprise; name: "Surprise"; axisX: xAxis; axisY: yAxis }
                    
                    function appendLimited(series, time, value) {
                        series.append(time, value);
                        if (series.count > maxPoints) {
                            series.remove(0);
                        }
                    }
                    
                    function addDataPoint(probabilities) {
                        appendLimited(seriesAnger, timeCounter, probabilities.anger);
                        appendLimited(seriesDisgust, timeCounter, probabilities.disgust);
                        appendLimited(seriesFear, timeCounter, probabilities.fear);
                        appendLimited(seriesHappiness, timeCounter, probabilities.happiness);
                        appendLimited(seriesNeutral, timeCounter, probabilities.neutral);
                        appendLimited(seriesSadness, timeCounter, probabilities.sadness);
                        appendLimited(seriesSurprise, timeCounter, probabilities.surprise);
                        
                        // اسکرول محور X
                        if (timeCounter >=  xAxis.max - 10) {
                            chartView.startTime += 2;
                            xAxis.min = chartView.startTime;
                            xAxis.max = chartView.startTime + chartView.maxPoints;
                        }
                        timeCounter += 1;  // زمان جلو بره
                        
                    }
                }
                
            }
            
            
            
            Rectangle{
                
                id:showEmotion
                width: parent.width / 4 + 20
                height: parent.height / 3.3
                color: "#424242"
                
                border.color: "#424242"
                border.width: 1.5
                
                anchors{
                    top: circulePlot.bottom
                    topMargin: 20
                    left: parent.left
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                
                Text {
                    id: emotionStateNowThree
                    text: "Emotion: " + emotionStatus
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: showEmotion.top
                        topMargin: 20
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                Text {
                    id: percentageNowThree
                    text: "Percentage: " +  Math.round(percentageEmotion*100) + "%"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: emotionStateNowThree.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                
                Text {
                    id: genderIdThree
                    text: "Gender: " + "Male"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: percentageNowThree.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                Button{
                    id:startStopIdThree
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdThree.bottom
                        topMargin: 40
                        left: parent.left
                        leftMargin: 20
                    }
                    
                    contentItem: Text {
                        text: !startStop? qsTr("Start") : qsTr("Stop")
                        color: "white"
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: {
                        if(!startStop){
                            startStop = !startStop
                            if (inputSourceCombo.currentText === "Video File") {
                                faceEmotionRecognation.processVideo()
                            } else {
                                faceEmotionRecognation.startWorker()
                            }
                        }
                        else{
                            if (inputSourceCombo.currentText === "Video File") {
                                faceEmotionRecognation.stopVideoProcessing()
                            } else {
                                faceEmotionRecognation.stopWorker()
                            }
                            startStop = !startStop
                        }
                        
                    }
                }
                
                Button{
                    id:backIdThree
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdThree.bottom
                        topMargin: 40
                        left: startStopIdThree.right
                        leftMargin: 5
                    }
                    
                    contentItem: Text {
                        text: qsTr("back")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: stackView.push(secondPage)
                }
                
                Button{
                    id:pdfIdThree
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdThree.bottom
                        topMargin: 40
                        left: backIdThree.right
                        leftMargin: 5
                    }
                    
                    contentItem: Text {
                        text: qsTr("PDF Report")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: faceEmotionRecognation.generatePdfReport()
                }
                
            }
            
            Rectangle{
                // مدل داده‌ها
                
                id:tableTimeEmotion
                width: parent.width / 3 - 100
                height: parent.height / 3.3
                color: "#424242"
                
                border.color: "#424242"
                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: showEmotion.right
                    leftMargin: 20
                }

                Column {
                    id:buttonMenu
                    spacing: 20
                    anchors {
                        bottom: parent.bottom
                        bottomMargin: 10
                        horizontalCenter: parent.horizontalCenter
                    }

                    ComboBox {
                        id: inputSourceCombo
                        width: 120
                        height: 40
                        model: ["Webcam", "Video File"]
                        currentIndex: 0
                        
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        
                        contentItem: Text {
                            text: inputSourceCombo.displayText
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }

                        onCurrentTextChanged: {
                            if (currentText === "Video File") {
                                selectVideoButton.visible = true
                                startStopIdThree.enabled = false
                            } else {
                                selectVideoButton.visible = false
                                startStopIdThree.enabled = true
                            }
                        }
                    }

                    ProgressBar {
                        id: videoProgressBar
                        width: 120
                        height: 20
                        visible: false
                        from: 0
                        to: 100
                        value: 0
                    }

                    Text {
                        id: processingStatusText
                        text: ""
                        color: "white"
                        visible: false
                    }

                    Button {
                        id: selectVideoButton
                        width: 120
                        height: 40
                        visible: false
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        contentItem: Text {
                            text: qsTr("Select Video")
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        onClicked: videoDialog.open()
                    }

                    Button {
                        id: openFormButton
                        width: 120
                        height: 40
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        contentItem: Text {
                            text: qsTr("Add Person Info")
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        onClicked: personInfoDialogThree.open()
                    }
                }

                FileDialog {
                    id: videoDialog
                    title: "Please choose a video file"
                    nameFilters: ["Video files (*.mp4 *.avi *.mkv)"]
                    onAccepted: {
                        console.log("Selected video:", selectedFile)
                        faceEmotionRecognation.setVideoPath(selectedFile)
                    }
                }

                Dialog {
                    id: personInfoDialogThree
                    
                    width: 300
                    height: 290
                    anchors.centerIn: parent
                    modal: true
                    standardButtons: Dialog.Ok | Dialog.Cancel

                    property alias nameField: nameInputThree.text
                    property alias lastNameField: lastNameInputThree.text
                    property alias ageField: ageInputThree.text
                    property alias nationalCodeField: nationalCodeInputThree.text

                    Column {
                        spacing: 20
                        anchors.fill: parent
                        anchors.margins: 10

                        TextField {
                            id: nameInputThree
                            placeholderText: "Enter name"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: lastNameInputThree
                            placeholderText: "Enter lastName"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: ageInputThree
                            placeholderText: "Enter age"
                            width: parent.width
                            color: "black"
                            validator: IntValidator {bottom: 0; top: 120;}
                            height:30
                        }

                        TextField {
                            id: nationalCodeInputThree
                            placeholderText: "Enter nationalCode"
                            width: parent.width
                            color: "black"
                            height:30
                        }
                    }

                    onAccepted: {
                        faceEmotionRecognation.savePersonInfo(nameField,lastNameField,ageField,nationalCodeField)
                    }

                    onRejected: {
                        nameInputThree.text = ""
                        lastNameInputThree.text = ""
                        ageInputThree.text = ""
                        nationalCodeInputThree.text = ""
                    }
                }
            }
            
            Rectangle{
                // مدل داده‌ها
                
                id: donatChartFour
                width: parent.width / 3 + 100
                height: parent.height / 3.3
                color: "#424242"
                
                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: tableTimeEmotion.right
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                ChartView {
                    id:donutParent
                    width: parent.width
                    height: parent.height
                    backgroundColor: "#424242"
                    legend.labelColor: "white" // رنگ نوشته‌های legend                        antialiasing: true
                    
                    // نمودار دایره‌ای
                    PieSeries {
                        id: pieSeries
                        holeSize: 0.5  // اندازه حفره برای ایجاد شکل دونات
                        
                        // احساسات مختلف و مقدارشان
                        PieSlice {id:angerSlice ; label: "Anger"; value: 0; color: "red" }
                        PieSlice {id:happinessSlice ; label: "Happiness"; value: 0; color: "green" }
                        PieSlice {id:fearSlice ; label: "Fear"; value: 0; color: "purple" }
                        PieSlice {id:sadnessSlice ; label: "Sadness"; value: 0; color: "blue" }
                        PieSlice {id:disgustSlice ; label: "Disgust"; value: 0; color: "yellow" }
                        PieSlice {id:surpriseSlice ; label: "Surprise"; value: 0; color: "pink" }
                        PieSlice {id:neutralSlice ; label: "Neutral"; value: 0; color: "orange" }
                    }
                    // تابع برای آپدیت مقدار احساسات در نمودار
                    function updateDonutChart(emotions) {
                        angerSlice.value = emotions["anger"]
                        happinessSlice.value = emotions["happiness"]
                        fearSlice.value    = emotions["fear"]
                        sadnessSlice.value  = emotions["sadness"]
                        disgustSlice.value  = emotions["disgust"]
                        surpriseSlice.value = emotions["surprise"]
                        neutralSlice.value  = emotions["neutral"]
                        
                    }
                }
                
                
            }
            Connections{
                target: faceEmotionRecognation
                function onEmotionRsult(emotion_status,emotion_prob,total_prob) {
                    emotionDataThree.emotions["anger"] = total_prob[0] * 100;
                    emotionDataThree.emotions["disgust"] = total_prob[1] * 100;
                    emotionDataThree.emotions["fear"] = total_prob[2] * 100;
                    emotionDataThree.emotions["happiness"] = total_prob[3] * 100;
                    emotionDataThree.emotions["neutral"] = total_prob[4] * 100;
                    emotionDataThree.emotions["sadness"] = total_prob[5] * 100;
                    emotionDataThree.emotions["surprise"] = total_prob[6] * 100;
                    percentageEmotion = emotion_prob;
                    emotionStatus = emotion_status;
                    var time = seriesAnger.count;
                    chartView.addDataPoint(emotionDataThree.emotions);
                    newEmotions[emotion_status] += 1;
                    donutParent.updateDonutChart(newEmotions);
                    emotionDataThree.emotionsQmlChanged();
                }
            }

            Connections {
                target: faceEmotionRecognation
                function onProcessingStatus(status, progress) {
                    videoProgressBar.visible = true
                    processingStatusText.visible = true
                    processingStatusText.text = status
                    videoProgressBar.value = progress
                    
                    if (progress >= 100 || progress <= 0) {
                        // Hide progress bar after 2 seconds when processing is complete
                        timer.start()
                    }
                }
            }

            Timer {
                id: timer
                interval: 2000
                repeat: false
                onTriggered: {
                    videoProgressBar.visible = false
                    processingStatusText.visible = false
                }
            }
        }
    }

    //four page voice
    Component {
        id: fourPage
        Rectangle {
            QtObject {
                id: emotionSpeachData
                property var emotionsSpeech: ({
                                                  "angry": 0,
                                                  "disgusted": 0,
                                                  "fearful": 0,
                                                  "happy": 0,
                                                  "neutral": 0,
                                                  "sad": 0,
                                                  "surprised": 0,
                                              });
                signal emotionsSpeachQmlChanged();
            }
            property var emotionsSpeech: ({
                                              "angry":0,
                                              "disgusted":0,
                                              "fearful":0,
                                              "happy":0,
                                              "neutral":0,
                                              "other":0,
                                              "sad":0,
                                              "surprised":0,
                                              "unknown":0
                                          })
            property var newEmotionsSpeech : {
                "angry":0,
                "disgusted":0,
                "fearful":0,
                "happy":0,
                "neutral":0,
                "other":0,
                "sad":0,
                "surprised":0,
                "unknown":0
            };
            
            property var emotionKeys: ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
            
            id:page3Id
            property bool startStopEmotion : false
            property double percentageEmotionSpeech: 0;
            property string emotionSpeechStatus: "Empty";
            anchors.fill: parent
            color: "#212121"
            Text {
                id: centerName
                text: qsTr("Real Time Speech Emotion Detection")
                font.family: "Arial"
                font.pointSize: 18
                font.bold:true
                color: titleText
                anchors{
                    top: page3Id.top
                    topMargin: 20
                    horizontalCenter: parent.horizontalCenter
                }
            }
            Rectangle{
                
                id:circulePlot
                width: parent.width / 4 + 20
                height: parent.height / 1.8
                color: "#424242"
                
                border.color: "#424242"
                border.width: 1.5
                
                anchors{
                    top: centerName.bottom
                    topMargin: 30
                    left: parent.left
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                
                Column {
                    width : parent.width - 10
                    height : parent.height -10
                    spacing: 1
                    Grid {
                        id: grid
                        columns: 2      // دو ستون
                        spacing: 2
                        //anchors.horizontalCenter: parent.horizontalCenter
                        Repeater {
                            model: emotionKeys
                            delegate: Item {
                                width: 180
                                height: 100
                                // تعیین رنگ هر احساس بر اساس کلید آن
                                property string emotionColor: {
                                    switch(modelData) {
                                    case "angry": return "red";
                                    case "disgusted": return "yellow";
                                    case "fearful": return "purple";
                                    case "happy": return "green";
                                    case "neutral": return "orange";
                                    case "sad": return "blue";
                                    case "surprised": return "pink";
                                    default: return "cyan";
                                    }
                                }
                                
                                Canvas {
                                    id:progressCanvas
                                    anchors.fill: parent
                                    Connections {
                                        target: emotionSpeachData
                                        onEmotionsSpeachQmlChanged: progressCanvas.requestPaint()
                                    }
                                    onPaint: {
                                        var ctx = getContext("2d");
                                        ctx.clearRect(0, 0, width, height);
                                        
                                        var percentage = emotionSpeachData.emotionsSpeech[modelData] || 0;  // درصد مربوط به احساس فعلی
                                        var startAngle = Math.PI;                // شروع از ۱۸۰ درجه (رادیان)
                                        var endAngle = Math.PI * (1 + (percentage / 100)); // انتهای قوس بر اساس درصد
                                        var centerX = width / 2;
                                        var centerY = height - 10;
                                        var radius = 50;
                                        
                                        // رسم نیم‌دایره پس‌زمینه (رنگ خاکستری)
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = "gray";
                                        ctx.stroke();
                                        
                                        // رسم نیم‌دایره پیشرفت (با رنگ اختصاصی)
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = emotionColor;
                                        ctx.stroke();
                                    }
                                }
                                
                                Text {
                                    id :textCircule;
                                    text: modelData;
                                    color: "white"
                                    anchors.horizontalCenter: parent.horizontalCenter;
                                    anchors.top: progressCanvas.bottom;
                                    font.bold: true;
                                    
                                    Connections {
                                        target: emotionSpeachData
                                        onEmotionsSpeachQmlChanged: textCircule.text = modelData + ": " + Math.round(emotionSpeachData.emotionsSpeech[modelData] || 0) + "%"
                                    }
                                }
                            }
                        }
                    }
                }
                
            }
            
            Rectangle {
                id: linePlot
                width: parent.width / 1.5 + 20
                height: parent.height / 1.8
                color: "#424242"
                border.color: "#424242"
                border.width: 1.5
                
                anchors {
                    top: centerName.bottom
                    topMargin: 30
                    left: circulePlot.right
                    leftMargin: 20
                }
                
                ChartView {
                    id: chartView
                    anchors.fill: parent
                    antialiasing: true
                    title: "Emotion Probabilities Over Time"
                    backgroundColor: "#424242"
                    legend.visible: true
                    legend.labelColor: "white"
                    titleColor: "white"
                    
                    property int maxPoints: 100
                    property real startTime: 0
                    property int timeCounter: 0  // زمان مستقل برای محور x
                    
                    ValueAxis {
                        id: xAxis
                        min: chartView.startTime
                        max: chartView.startTime + chartView.maxPoints
                        tickCount: 11
                        titleText: "Time"
                        labelsColor: "white"
                    }
                    
                    ValueAxis {
                        id: yAxis
                        min: 0
                        max: 100
                        tickCount: 11
                        titleText: "Probability (%)"
                        labelsColor: "white"
                    }
                    
                    // سری‌ها
                    LineSeries { id: seriesAnger; name: "angry"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesDisgust; name: "disgusted"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesFear; name: "fearful"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesHappiness; name: "happy"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesNeutral; name: "neutral"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSadness; name: "sad"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSurprise; name: "surprised"; axisX: xAxis; axisY: yAxis }
                    
                    function appendLimited(series, time, value) {
                        series.append(time, value);
                        if (series.count > maxPoints) {
                            series.remove(0);
                        }
                    }
                    
                    function addDataPoint(probabilities) {
                        appendLimited(seriesAnger, timeCounter, probabilities.angry);
                        appendLimited(seriesDisgust, timeCounter, probabilities.disgusted);
                        appendLimited(seriesFear, timeCounter, probabilities.fearful);
                        appendLimited(seriesHappiness, timeCounter, probabilities.happy);
                        appendLimited(seriesNeutral, timeCounter, probabilities.neutral);
                        appendLimited(seriesSadness, timeCounter, probabilities.sad);
                        appendLimited(seriesSurprise, timeCounter, probabilities.surprised);
                        
                        // اسکرول محور X
                        if (timeCounter >=  xAxis.max - 10) {
                            chartView.startTime += 2;
                            xAxis.min = chartView.startTime;
                            xAxis.max = chartView.startTime + chartView.maxPoints;
                        }
                        timeCounter += 1;  // زمان جلو بره
                        
                    }
                }
                
            }
                       
            Rectangle{
                
                id:showEmotion
                width: parent.width / 4 + 20
                height: parent.height / 3.3
                color: "#424242"
                
                border.color: "#424242"
                border.width: 1.5
                
                anchors{
                    top: circulePlot.bottom
                    topMargin: 20
                    left: parent.left
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                
                Text {
                    id: emotionStateNow
                    text: "Emotion: " + emotionSpeechStatus
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: showEmotion.top
                        topMargin: 20
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                Text {
                    id: percentageNow
                    text: "Percentage: " +  Math.round(percentageEmotionSpeech*100) + "%"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: emotionStateNow.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                
                Text {
                    id: genderId
                    text: "Gender: " + "Male"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: percentageNow.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                Button{
                    id:startStopId
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderId.bottom
                        topMargin: 40
                        left: parent.left
                        leftMargin: 20
                    }
                    
                    contentItem: Text {
                        text: !startStopEmotion? qsTr("Start") : qsTr("Stop")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: {
                        if(!startStopEmotion){
                            startStopEmotion = !startStopEmotion
                            backEnd.startWorker()
                        }
                        else{
                            backEnd.stopWorker()
                            startStopEmotion = !startStopEmotion
                        }
                        
                    }
                }
                
                Button{
                    id:backId
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderId.bottom
                        topMargin: 40
                        left: startStopId.right
                        leftMargin: 5
                    }
                    
                    contentItem: Text {
                        text: qsTr("back")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: stackView.push(secondPage)
                }
                
                Button{
                    id:pdfId
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderId.bottom
                        topMargin: 40
                        left: backId.right
                        leftMargin: 5
                    }
                    
                    contentItem: Text {
                        text: qsTr("PDF Report")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: backEnd.generatePdfReport()
                }
                
            }
            
            Rectangle{
                // مدل داده‌ها
                
                id:tableTimeEmotion
                width: parent.width / 3 - 100
                height: parent.height / 3.3
                color: "#424242"
                
                border.color: "#424242"
                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: showEmotion.right
                    leftMargin: 20
                }

                Column {
                    spacing: 10
                    anchors {
                        bottom: parent.bottom
                        bottomMargin: 10
                        horizontalCenter: parent.horizontalCenter
                    }

                    ComboBox {
                        id: audioInputSourceCombo
                        width: 120
                        height: 40
                        model: ["Microphone", "Audio File"]
                        currentIndex: 0
                        
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        
                        contentItem: Text {
                            text: audioInputSourceCombo.displayText
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        
                        delegate: ItemDelegate {
                            width: audioInputSourceCombo.width
                            contentItem: Text {
                                text: modelData
                                color: "black"
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            highlighted: audioInputSourceCombo.highlightedIndex === index
                        }

                        onCurrentTextChanged: {
                            if (currentText === "Audio File") {
                                selectAudioButton.visible = true
                                startStopId.enabled = false
                            } else {
                                selectAudioButton.visible = false
                                startStopId.enabled = true
                                // Reset به حالت میکروفن
                                backEnd.resetToMicrophoneMode()
                            }
                        }
                    }
                    
                    Button {
                        id: selectAudioButton
                        width: 120
                        height: 40
                        visible: false
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        contentItem: Text {
                            text: qsTr("Select Audio")
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        onClicked: audioDialog.open()
                    }

                    Button {
                        id: openFormButton
                        width: 120
                        height: 40
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        contentItem: Text {
                            text: qsTr("Add Person Info")
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        onClicked: personInfoDialogThree.open()
                    }
                }

                Dialog {
                    id: personInfoDialogThree
                    width: 300
                    height: 290
                    anchors.centerIn: parent
                    modal: true
                    standardButtons: Dialog.Ok | Dialog.Cancel

                    property alias nameField: nameInputThree.text
                    property alias lastNameField: lastNameInputThree.text
                    property alias ageField: ageInputThree.text
                    property alias nationalCodeField: nationalCodeInputThree.text

                    Column {
                        spacing: 20
                        anchors.fill: parent
                        anchors.margins: 10

                        TextField {
                            id: nameInputThree
                            placeholderText: "Enter name"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: lastNameInputThree
                            placeholderText: "Enter lastName"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: ageInputThree
                            placeholderText: "Enter age"
                            width: parent.width
                            color: "black"
                            validator: IntValidator {bottom: 0; top: 120;}
                            height:30
                        }

                        TextField {
                            id: nationalCodeInputThree
                            placeholderText: "Enter nationalCode"
                            width: parent.width
                            color: "black"
                            height:30
                        }
                    }

                    onAccepted: {
                        faceEmotionRecognation.savePersonInfo(nameField,lastNameField,ageField,nationalCodeField)
                    }

                    onRejected: {
                        nameInputThree.text = ""
                        lastNameInputThree.text = ""
                        ageInputThree.text = ""
                        nationalCodeInputThree.text = ""
                    }
                }
            }
            
            Rectangle{
                // مدل داده‌ها
                
                id: donatChart
                width: parent.width / 3 + 100
                height: parent.height / 3.3
                color: "#424242"
                
                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: tableTimeEmotion.right
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                ChartView {
                    id:donutParent
                    width: parent.width
                    height: parent.height
                    backgroundColor: "#424242"
                    legend.labelColor: "white" // رنگ نوشته‌های legend                        antialiasing: true
                    
                    // نمودار دایره‌ای
                    PieSeries {
                        id: pieSeries
                        holeSize: 0.5  // اندازه حفره برای ایجاد شکل دونات
                        
                        // احساسات مختلف و مقدارشان
                        PieSlice {id:angerSlice ; label: "Anger"; value: 0; color: "red" }
                        PieSlice {id:happinessSlice ; label: "Happiness"; value: 0; color: "green" }
                        PieSlice {id:fearSlice ; label: "Fear"; value: 0; color: "purple" }
                        PieSlice {id:sadnessSlice ; label: "Sadness"; value: 0; color: "blue" }
                        PieSlice {id:disgustSlice ; label: "Disgust"; value: 0; color: "yellow" }
                        PieSlice {id:surpriseSlice ; label: "Surprise"; value: 0; color: "pink" }
                        PieSlice {id:neutralSlice ; label: "Neutral"; value: 0; color: "orange" }
                    }
                    // تابع برای آپدیت مقدار احساسات در نمودار
                    function updateDonutChart(emotions) {
                        angerSlice.value = emotions["angry"]
                        happinessSlice.value = emotions["happy"]
                        fearSlice.value    = emotions["fearful"]
                        sadnessSlice.value  = emotions["sad"]
                        disgustSlice.value  = emotions["disgusted"]
                        surpriseSlice.value = emotions["surprised"]
                        neutralSlice.value  = emotions["neutral"]
                        
                    }
                }
                
                
            }
            
            FileDialog {
                id: audioDialog
                title: "Please choose an audio file"
                nameFilters: ["Audio files (*.wav *.mp3 *.m4a *.flac *.ogg)"]
                onAccepted: {
                    console.log("Selected audio:", selectedFile)
                    backEnd.setAudioPath(selectedFile)
                }
            }
            
            Connections{
                target: backEnd
                function onDataUpdated(emotion_status,emotion_prob,total_prob) {
                    emotionSpeachData.emotionsSpeech["angry"] = total_prob[0] * 100;
                    emotionSpeachData.emotionsSpeech["disgusted"] = total_prob[1] * 100;
                    emotionSpeachData.emotionsSpeech["fearful"] = total_prob[2] * 100;
                    emotionSpeachData.emotionsSpeech["happy"] = total_prob[3] * 100;
                    emotionSpeachData.emotionsSpeech["neutral"] = total_prob[4] * 100;
                    emotionSpeachData.emotionsSpeech["sad"] = total_prob[6] * 100;
                    emotionSpeachData.emotionsSpeech["surprised"] = total_prob[7] * 100;
                    percentageEmotionSpeech = emotion_prob;
                    emotionSpeechStatus = emotion_status;
                    chartView.addDataPoint(emotionSpeachData.emotionsSpeech);
                    newEmotionsSpeech[emotion_status] += 1;
                    donutParent.updateDonutChart(newEmotionsSpeech);
                    emotionSpeachData.emotionsSpeachQmlChanged();
                }
            }
        }
    }

    //five page fution
    Component {
        id: fivePage
        Rectangle {
            QtObject {
                id: emotionDataFive
                property var emotionsFusion: ({
                    "anger": 0,
                    "disgust": 0,
                    "fear": 0,
                    "happiness": 0,
                    "neutral": 0,
                    "sadness": 0,
                    "surprise": 0
                });
                signal emotionsFusionQmlChanged();
            }

            property var newEmotionsFusion : {
                "anger": 0,
                "disgust": 0,
                "fear": 0,
                "happiness": 0,
                "neutral": 0,
                "sadness": 0,
                "surprise": 0
            }

            property var emotionKeys: ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

            Connections {
                target: backEndFution
                function onFinalResult(emotion_status, emotion_prob, total_prob) {
                    emotionDataFive.emotionsFusion["anger"] = total_prob[0] * 100;
                    emotionDataFive.emotionsFusion["disgust"] = total_prob[1] * 100;
                    emotionDataFive.emotionsFusion["fear"] = total_prob[2] * 100;
                    emotionDataFive.emotionsFusion["happiness"] = total_prob[3] * 100;
                    emotionDataFive.emotionsFusion["neutral"] = total_prob[4] * 100;
                    emotionDataFive.emotionsFusion["sadness"] = total_prob[5] * 100;
                    emotionDataFive.emotionsFusion["surprise"] = total_prob[6] * 100;
                    percentageEmotionFusion = emotion_prob;
                    emotionFusionStatus = emotion_status;
                    chartView.addDataPoint(emotionDataFive.emotionsFusion);
                    newEmotionsFusion[emotion_status] += 1;
                    donutParent.updateDonutChart(newEmotionsFusion);
                    emotionDataFive.emotionsFusionQmlChanged();
                }
            }

            id:page5Id
            property bool startStopfution : false
            property double percentageEmotionFusion: 0;
            property string emotionFusionStatus: "Empty";
            anchors.fill: parent
            color: "#212121"
            Text {
                id: centerName
                text: qsTr("Real Time Fusion Emotion Detection")
                font.family: "Arial"
                font.pointSize: 18
                font.bold:true
                color: titleText
                anchors{
                    top: page5Id.top
                    topMargin: 20
                    horizontalCenter: parent.horizontalCenter
                }
            }
            
            Rectangle{
                id:circulePlot
                width: parent.width / 4 + 20
                height: parent.height / 1.8
                color: "#424242"
                border.color: "#424242"
                border.width: 1.5
                
                anchors{
                    top: centerName.bottom
                    topMargin: 30
                    left: parent.left
                    leftMargin: 20
                }
                
                Column {
                    width : parent.width - 10
                    height : parent.height -10
                    spacing: 1
                    Grid {
                        id: gridFive
                        columns: 2
                        spacing: 2
                        Repeater {
                            model: emotionKeys
                            delegate: Item {
                                width: 180
                                height: 100
                                property string emotionColor: {
                                    switch(modelData) {
                                    case "anger": return "red";
                                    case "disgust": return "yellow";
                                    case "fear": return "purple";
                                    case "happiness": return "green";
                                    case "neutral": return "orange";
                                    case "sadness": return "blue";
                                    case "surprise": return "pink";
                                    default: return "cyan";
                                    }
                                }
                                
                                Canvas {
                                    id:progressCanvasFive
                                    anchors.fill: parent
                                    
                                    Connections {
                                        target: emotionDataFive
                                        onEmotionsFusionQmlChanged: progressCanvasFive.requestPaint()
                                    }
                                    onPaint: {
                                        var ctx = getContext("2d");
                                        ctx.clearRect(0, 0, width, height);
                                        
                                        var percentage = emotionDataFive.emotionsFusion[modelData] || 0;
                                        var startAngle = Math.PI;
                                        var endAngle = Math.PI * (1 + (percentage / 100));
                                        var centerX = width / 2;
                                        var centerY = height - 10;
                                        var radius = 50;
                                        
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = "gray";
                                        ctx.stroke();
                                        
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = emotionColor;
                                        ctx.stroke();
                                    }
                                }
                                
                                Text {
                                    id: textCircule
                                    text: modelData
                                    color: "white"
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.top: progressCanvasFive.bottom
                                    font.bold: true
                                    
                                    Connections {
                                        target: emotionDataFive
                                        onEmotionsFusionQmlChanged: textCircule.text = modelData + ": " + Math.round(emotionDataFive.emotionsFusion[modelData] || 0) + "%"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            Rectangle {
                id: linePlot
                width: parent.width / 1.5 + 20
                height: parent.height / 1.8
                color: "#424242"
                border.color: "#424242"
                border.width: 1.5
                
                anchors {
                    top: centerName.bottom
                    topMargin: 30
                    left: circulePlot.right
                    leftMargin: 20
                }
                
                ChartView {
                    id: chartView
                    anchors.fill: parent
                    antialiasing: true
                    title: "Emotion Probabilities Over Time"
                    backgroundColor: "#424242"
                    legend.visible: true
                    legend.labelColor: "white"
                    titleColor: "white"
                    
                    property int maxPoints: 100
                    property real startTime: 0
                    property int timeCounter: 0
                    
                    ValueAxis {
                        id: xAxis
                        min: chartView.startTime
                        max: chartView.startTime + chartView.maxPoints
                        tickCount: 11
                        titleText: "Time"
                        labelsColor: "white"
                    }
                    
                    ValueAxis {
                        id: yAxis
                        min: 0
                        max: 100
                        tickCount: 11
                        titleText: "Probability (%)"
                        labelsColor: "white"
                    }
                    
                    LineSeries { id: seriesAnger; name: "Anger"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesDisgust; name: "Disgust"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesFear; name: "Fear"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesHappiness; name: "Happiness"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesNeutral; name: "Neutral"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSadness; name: "Sadness"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSurprise; name: "Surprise"; axisX: xAxis; axisY: yAxis }
                    
                    function appendLimited(series, time, value) {
                        series.append(time, value);
                        if (series.count > maxPoints) {
                            series.remove(0);
                        }
                    }
                    
                    function addDataPoint(probabilities) {
                        appendLimited(seriesAnger, timeCounter, probabilities.anger);
                        appendLimited(seriesDisgust, timeCounter, probabilities.disgust);
                        appendLimited(seriesFear, timeCounter, probabilities.fear);
                        appendLimited(seriesHappiness, timeCounter, probabilities.happiness);
                        appendLimited(seriesNeutral, timeCounter, probabilities.neutral);
                        appendLimited(seriesSadness, timeCounter, probabilities.sadness);
                        appendLimited(seriesSurprise, timeCounter, probabilities.surprise);
                        
                        if (timeCounter >= xAxis.max - 10) {
                            chartView.startTime += 2;
                            xAxis.min = chartView.startTime;
                            xAxis.max = chartView.startTime + chartView.maxPoints;
                        }
                        timeCounter += 1;
                    }
                }
            }
            
            Rectangle{
                id:showEmotion
                width: parent.width / 4 + 20
                height: parent.height / 3.3
                color: "#424242"
                border.color: "#424242"
                border.width: 1.5
                
                anchors{
                    top: circulePlot.bottom
                    topMargin: 20
                    left: parent.left
                    leftMargin: 20
                }
                
                Text {
                    id: emotionStateNowFive
                    text: "Emotion: " + emotionFusionStatus
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: showEmotion.top
                        topMargin: 20
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                
                Text {
                    id: percentageNowFive
                    text: "Percentage: " + Math.round(percentageEmotionFusion*100) + "%"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: emotionStateNowFive.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                
                Text {
                    id: genderIdFive
                    text: "Gender: " + "Male"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: percentageNowFive.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                
                Button{
                    id:startStopIdFive
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdFive.bottom
                        topMargin: 40
                        left: parent.left
                        leftMargin: 20
                    }
                    
                    contentItem: Text {
                        text: !startStopfution? qsTr("Start") : qsTr("Stop")
                        color: "white"
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: {
                        if(!startStopfution){
                            startStopfution = !startStopfution
                            backEndFution.startWorker()
                        }
                        else{
                            backEndFution.stopWorker()
                            startStopfution = !startStopfution
                        }
                    }
                }
                
                Button{
                    id:backIdFive
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdFive.bottom
                        topMargin: 40
                        left: startStopIdFive.right
                        leftMargin: 5
                    }
                    
                    contentItem: Text {
                        text: qsTr("back")
                        color: "white"
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: stackView.push(secondPage)
                }
                
                Button{
                    id:pdfIdFive
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdFive.bottom
                        topMargin: 40
                        left: backIdFive.right
                        leftMargin: 5
                    }
                    
                    contentItem: Text {
                        text: qsTr("PDF Report")
                        color: "white"
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: backEndFution.generatePdfReport()
                }
                
            }
            
            Rectangle{
                id:tableTimeEmotion
                width: parent.width / 3 - 100
                height: parent.height / 3.3
                color: "#424242"
                
                border.color: "#424242"
                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: showEmotion.right
                    leftMargin: 20
                }

                Column {
                    id:buttonMenu
                    spacing: 1
                    anchors {
                        bottom: parent.bottom
                        bottomMargin: 5
                        horizontalCenter: parent.horizontalCenter
                    }

                    Text {
                        text: "Alpha (Face Weight): " + alphaSlider.value.toFixed(2)
                        color: "white"
                        font.pixelSize: 14
                    }

                    Slider {
                        id: alphaSlider
                        width: 200
                        from: 0.0
                        to: 1.0
                        value: 0.6
                        stepSize: 0.01
                        onValueChanged: {
                            backEndFution.setWeights(alphaSlider.value, betaSlider.value)
                        }
                    }

                    Text {
                        text: "Beta (Voice Weight): " + betaSlider.value.toFixed(2)
                        color: "white"
                        font.pixelSize: 14
                    }

                    Slider {
                        id: betaSlider
                        width: 200
                        from: 0.0
                        to: 1.0
                        value: 0.4
                        stepSize: 0.01
                        onValueChanged: {
                            backEndFution.setWeights(alphaSlider.value, betaSlider.value)
                        }
                    }

                    ComboBox {
                        id: emotionTypeCombo
                        width: 120
                        height: 40
                        model: ["Face&voice", "Face&EEG", "Voice&EEG", "All"]
                        currentIndex: 0
                        
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        
                        contentItem: Text {
                            text: emotionTypeCombo.displayText
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        
                        delegate: ItemDelegate {
                            width: emotionTypeCombo.width
                            contentItem: Text {
                                text: modelData
                                color: "black"
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            highlighted: emotionTypeCombo.highlightedIndex === index
                        }
                    }

                    

                    Button {
                        id: openFormButton
                        width: 120
                        height: 40
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        contentItem: Text {
                            text: qsTr("Add Person Info")
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        onClicked: personInfoDialogFive.open()
                    }
                }

                Dialog {
                    id: personInfoDialogFive
                    width: 300
                    height: 290
                    anchors.centerIn: parent
                    modal: true
                    standardButtons: Dialog.Ok | Dialog.Cancel

                    property alias nameField: nameInputFive.text
                    property alias lastNameField: lastNameInputFive.text
                    property alias ageField: ageInputFive.text
                    property alias nationalCodeField: nationalCodeInputFive.text

                    Column {
                        spacing: 20
                        anchors.fill: parent
                        anchors.margins: 10

                        TextField {
                            id: nameInputFive
                            placeholderText: "Enter name"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: lastNameInputFive
                            placeholderText: "Enter lastName"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: ageInputFive
                            placeholderText: "Enter age"
                            width: parent.width
                            color: "black"
                            validator: IntValidator {bottom: 0; top: 120;}
                            height:30
                        }

                        TextField {
                            id: nationalCodeInputFive
                            placeholderText: "Enter nationalCode"
                            width: parent.width
                            color: "black"
                            validator: IntValidator {bottom: 0; top: 120;}
                            height:30
                        }
                    }

                    onAccepted: {
                        console.log("Name:", nameField)
                        console.log("lastName:", lastNameField)
                        console.log("Age:", ageField)
                        console.log("NationalCode:", nationalCodeField)
                        fileHandler.savePersonInfo(nameField, lastNameField, parseInt(ageField), nationalCodeField)
                    }

                    onRejected: {
                        nameInputFive.text = ""
                        lastNameInputFive.text = ""
                        ageInputFive.text = ""
                        nationalCodeInputFive.text = ""
                    }
                }

                FileDialog {
                    id: fileDialog
                    title: "Please choose an EEG file"
                    nameFilters: ["EEG files (*.edf)"]
                    onAccepted: {
                        console.log("Selected file:", selectedFile)
                        fileHandler.sendPath(selectedFile)
                    }
                }
            }
            
            Rectangle{
                    id: donatChartThree
                    width: parent.width / 3 + 100
                    height: parent.height / 3.3
                    color: "#424242"
                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: tableTimeEmotion.right
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                
                ChartView {
                    id:donutParent
                    width: parent.width
                    height: parent.height
                    backgroundColor: "#424242"
                    legend.labelColor: "white" // رنگ نوشته‌های legend                        antialiasing: true

                    // نمودار دایره‌ای
                    PieSeries {
                        id: pieSeries
                        holeSize: 0.5
                        
                        PieSlice {id:angerSlice ; label: "Anger"; value: 0; color: "red" }
                        PieSlice {id:happinessSlice ; label: "Happiness"; value: 0; color: "green" }
                        PieSlice {id:fearSlice ; label: "Fear"; value: 0; color: "purple" }
                        PieSlice {id:sadnessSlice ; label: "Sadness"; value: 0; color: "blue" }
                        PieSlice {id:disgustSlice ; label: "Disgust"; value: 0; color: "yellow" }
                        PieSlice {id:surpriseSlice ; label: "Surprise"; value: 0; color: "pink" }
                        PieSlice {id:neutralSlice ; label: "Neutral"; value: 0; color: "orange" }
                    }
                    
                    function updateDonutChart(emotions) {
                        angerSlice.value = emotions["anger"]
                        happinessSlice.value = emotions["happiness"]
                        fearSlice.value = emotions["fear"]
                        sadnessSlice.value = emotions["sadness"]
                        disgustSlice.value = emotions["disgust"]
                        surpriseSlice.value = emotions["surprise"]
                        neutralSlice.value = emotions["neutral"]
                    }
                }
            }
            Connections{
                target: backEndFution
                function onFinalResult(emotion_status,emotion_prob,total_prob) {
                    emotionDataFive.emotionsFusion["anger"] = total_prob[0] * 100;
                    emotionDataFive.emotionsFusion["disgust"] = total_prob[1] * 100;
                    emotionDataFive.emotionsFusion["fear"] = total_prob[2] * 100;
                    emotionDataFive.emotionsFusion["happiness"] = total_prob[3] * 100;
                    emotionDataFive.emotionsFusion["neutral"] = total_prob[4] * 100;
                    emotionDataFive.emotionsFusion["sadness"] = total_prob[5] * 100;
                    emotionDataFive.emotionsFusion["surprise"] = total_prob[6] * 100;
                    percentageEmotionFusion = emotion_prob;
                    emotionFusionStatus = emotion_status;
                    chartView.addDataPoint(emotionDataFive.emotionsFusion);
                    newEmotionsFusion[emotion_status] += 1;
                    donutParent.updateDonutChart(newEmotionsFusion);
                    emotionDataFive.emotionsFusionQmlChanged();
                }
            }
        }
    }

    //six page EEG
    Component {
        id: sixPage
        Rectangle {
            QtObject {
                id: emotionDataSix
                property var emotions: ({
                                            "anger": 0,
                                            "disgust": 0,
                                            "fear": 0,
                                            "happiness": 0,
                                            "neutral": 0,
                                            "sadness": 0,
                                            "surprise": 0,
                                        });
                signal emotionsQmlChanged();
            }
            property var newEmotions : {
                "anger": 0,
                "disgust": 0,
                "fear": 0,
                "happiness": 0,
                "neutral": 0,
                "sadness": 0,
                "surprise": 0,
            };

            property var emotionKeys: ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

            id:page6Id
            property bool startStop : false
            property double percentageEmotion: 0;
            property string emotionStatus: "Empty";
            anchors.fill: parent
            color: "#212121"
            Text {
                id: centerName
                text: qsTr("EEG Emotion Detection")
                font.family: "Arial"
                font.pointSize: 18
                font.bold:true
                color: titleText
                anchors{
                    top: page6Id.top
                    topMargin: 20
                    horizontalCenter: parent.horizontalCenter
                }
            }

            Rectangle{

                id:circulePlot
                width: parent.width / 4 + 20
                height: parent.height / 1.8
                color: "#424242"

                border.color: "#424242"
                border.width: 1.5

                anchors{
                    top: centerName.bottom
                    topMargin: 30
                    left: parent.left
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }

                Column {
                    width : parent.width - 10
                    height : parent.height -10
                    spacing: 1
                    Grid {
                        id: gridSix
                        columns: 2      // دو ستون
                        spacing: 2
                        //anchors.horizontalCenter: parent.horizontalCenter
                        Repeater {
                            model: emotionKeys
                            delegate: Item {
                                width: 180
                                height: 100
                                // تعیین رنگ هر احساس بر اساس کلید آن
                                property string emotionColor: {
                                    switch(modelData) {
                                    case "anger": return "red";
                                    case "disgust": return "yellow";
                                    case "fear": return "purple";
                                    case "happiness": return "green";
                                    case "neutral": return "orange";
                                    case "sadness": return "blue";
                                    case "surprise": return "pink";
                                    default: return "cyan";
                                    }
                                }

                                Canvas {
                                    id:progressCanvasSix
                                    anchors.fill: parent

                                    Connections {
                                        target: emotionDataSix
                                        onEmotionsQmlChanged: progressCanvasSix.requestPaint()
                                    }
                                    onPaint: {
                                        var ctx = getContext("2d");
                                        ctx.clearRect(0, 0, width, height);

                                        var percentage = emotionDataSix.emotions[modelData] || 0;  // درصد مربوط به احساس فعلی
                                        var startAngle = Math.PI;                // شروع از ۱۸۰ درجه (رادیان)
                                        var endAngle = Math.PI * (1 + (percentage / 100)); // انتهای قوس بر اساس درصد
                                        var centerX = width / 2;
                                        var centerY = height - 10;
                                        var radius = 50;

                                        // رسم نیم‌دایره پس‌زمینه (رنگ خاکستری)
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = "gray";
                                        ctx.stroke();

                                        // رسم نیم‌دایره پیشرفت (با رنگ اختصاصی)
                                        ctx.beginPath();
                                        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
                                        ctx.lineWidth = 10;
                                        ctx.strokeStyle = emotionColor;
                                        ctx.stroke();
                                    }
                                }

                                Text {
                                    id :textCircule;
                                    text: modelData + ": ";
                                    color: "white"
                                    anchors.horizontalCenter: parent.horizontalCenter;
                                    anchors.top: progressCanvasSix.bottom;
                                    font.bold: true;
                                    Connections {
                                        target: emotionDataSix
                                        onEmotionsQmlChanged: textCircule.text = modelData + ": " + Math.round(emotionDataSix.emotions[modelData] || 0) + "%"
                                    }
                                }
                            }
                        }
                    }
                }

            }

            Rectangle {
                id: linePlot
                width: parent.width / 1.5 + 20
                height: parent.height / 1.8
                color: "#424242"
                border.color: "#424242"
                border.width: 1.5

                anchors {
                    top: centerName.bottom
                    topMargin: 30
                    left: circulePlot.right
                    leftMargin: 20
                }

                ChartView {
                    id: chartView
                    anchors.fill: parent
                    antialiasing: true
                    title: "Emotion Probabilities Over Time"
                    backgroundColor: "#424242"
                    legend.visible: true
                    legend.labelColor: "white"
                    titleColor: "white"

                    property int maxPoints: 100
                    property real startTime: 0
                    property int timeCounter: 0  // زمان مستقل برای محور x

                    ValueAxis {
                        id: xAxis
                        min: chartView.startTime
                        max: chartView.startTime + chartView.maxPoints
                        tickCount: 11
                        titleText: "Time"
                        labelsColor: "white"
                    }

                    ValueAxis {
                        id: yAxis
                        min: 0
                        max: 100
                        tickCount: 11
                        titleText: "Probability (%)"
                        labelsColor: "white"
                    }

                    // سری‌ها
                    LineSeries { id: seriesAnger; name: "Anger"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesDisgust; name: "Disgust"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesFear; name: "Fear"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesHappiness; name: "Happiness"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesNeutral; name: "Neutral"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSadness; name: "Sadness"; axisX: xAxis; axisY: yAxis }
                    LineSeries { id: seriesSurprise; name: "Surprise"; axisX: xAxis; axisY: yAxis }

                    function appendLimited(series, time, value) {
                        series.append(time, value);
                        if (series.count > maxPoints) {
                            series.remove(0);
                        }
                    }

                    function addDataPoint(probabilities) {
                        appendLimited(seriesAnger, timeCounter, probabilities.anger);
                        appendLimited(seriesDisgust, timeCounter, probabilities.disgust);
                        appendLimited(seriesFear, timeCounter, probabilities.fear);
                        appendLimited(seriesHappiness, timeCounter, probabilities.happiness);
                        appendLimited(seriesNeutral, timeCounter, probabilities.neutral);
                        appendLimited(seriesSadness, timeCounter, probabilities.sadness);
                        appendLimited(seriesSurprise, timeCounter, probabilities.surprise);

                        // اسکرول محور X
                        if (timeCounter >=  xAxis.max - 10) {
                            chartView.startTime += 2;
                            xAxis.min = chartView.startTime;
                            xAxis.max = chartView.startTime + chartView.maxPoints;
                        }
                        timeCounter += 1;  // زمان جلو بره

                    }
                }

            }



            Rectangle{

                id:showEmotion
                width: parent.width / 4 + 20
                height: parent.height / 3.3
                color: "#424242"

                border.color: "#424242"
                border.width: 1.5

                anchors{
                    top: circulePlot.bottom
                    topMargin: 20
                    left: parent.left
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }

                Text {
                    id: emotionStateNowSix
                    text: "Emotion: " + emotionStatus
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: showEmotion.top
                        topMargin: 20
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                Text {
                    id: percentageNowSix
                    text: "Percentage: " +  Math.round(percentageEmotion*100) + "%"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: emotionStateNowSix.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }

                Text {
                    id: genderIdSix
                    text: "Gender: " + "Male"
                    font.family: "Arial"
                    font.pointSize: 18
                    font.bold:true
                    color: "white"
                    anchors{
                        top: percentageNowSix.top
                        topMargin: 50
                        left: showEmotion.left
                        leftMargin: 20
                    }
                }
                Button{
                    id:startStopIdSix
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdSix.bottom
                        topMargin: 40
                        left: parent.left
                        leftMargin: 20
                    }

                    contentItem: Text {
                        text: !startStop? qsTr("Start") : qsTr("Stop")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: {
                        if(!startStop){
                            startStop = !startStop
                            faceEmotionRecognation.startWorker()
                        }
                        else{
                            faceEmotionRecognation.stopWorker()
                            startStop = !startStop
                        }

                    }
                }

                Button{
                    id:backIdSix
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdSix.bottom
                        topMargin: 40
                        left: startStopIdSix.right
                        leftMargin: 5
                    }

                    contentItem: Text {
                        text: qsTr("back")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: stackView.push(secondPage)
                }

                Button{
                    id:pdfIdSix
                    width : 100
                    height : 40
                    background: Rectangle{
                        color: "gray"
                        radius: 10
                    }
                    anchors{
                        top: genderIdSix.bottom
                        topMargin: 40
                        left: backIdSix.right
                        leftMargin: 5
                    }

                    contentItem: Text {
                        text: qsTr("PDF Report")
                        color: "white" // رنگ متن
                        font.pixelSize: parent.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: fileHandler.generatePdfReport()
                }

            }
             Rectangle{
                id:tableTimeEmotion
                width: parent.width / 3 - 100
                height: parent.height / 3.3
                color: "#424242"
                border.color: "#424242"
                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: showEmotion.right
                    leftMargin: 20
                }

                Column {
                    spacing: 20
                    anchors {
                        bottom: parent.bottom
                        bottomMargin: 10
                        horizontalCenter: parent.horizontalCenter
                    }

                    ComboBox {
                        id: featureLevelCombo
                        width: 120
                        height: 40
                        model: ["Feature Level","Cheap", "All", "Moderate"]
                        currentIndex: 0
                        
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        
                        contentItem: Text {
                            text: featureLevelCombo.displayText
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        
                        delegate: ItemDelegate {
                            width: featureLevelCombo.width
                            contentItem: Text {
                                text: modelData
                                color: "black"
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            highlighted: featureLevelCombo.highlightedIndex === index
                        }

                        onCurrentTextChanged: {
                            fileHandler.setFeatureLevel(featureLevelCombo.currentText.toLowerCase())
                        }
                    }

                    Button {
                        id: selectFileButton
                        width: 120
                        height: 40
                        background: Rectangle {
                            color: "gray"
                            radius: 10
                        }
                        contentItem: Text {
                            text: qsTr("Select EEG File")
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        onClicked: fileDialogSix.open()
                    }
                }

                FileDialog {
                    id: fileDialogSix
                    title: "Please choose an EEG file"
                    nameFilters: ["EEG files (*.edf)"]
                    onAccepted: {
                        fileHandler.sendPath(currentFile)
                    }
                }

                Dialog {
                    id: personInfoDialogSix
                    width: 300
                    height: 290
                    anchors.centerIn: parent
                    modal: true
                    standardButtons: Dialog.Ok | Dialog.Cancel

                    property alias nameField: nameInputSix.text
                    property alias lastNameField: lastNameInputSix.text
                    property alias ageField: ageInputSix.text
                    property alias nationalCodeField: nationalCodeInputSix.text

                    Column {
                        spacing: 20
                        anchors.fill: parent
                        anchors.margins: 10

                        TextField {
                            id: nameInputSix
                            placeholderText: "Enter name"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: lastNameInputSix
                            placeholderText: "Enter lastName"
                            width: parent.width
                            color: "black"
                            height:30
                        }

                        TextField {
                            id: ageInputSix
                            placeholderText: "Enter age"
                            width: parent.width
                            color: "black"
                            validator: IntValidator {bottom: 0; top: 120;}
                            height:30
                        }

                        TextField {
                            id: nationalCodeInputSix
                            placeholderText: "Enter nationalCode"
                            width: parent.width
                            color: "black"
                            validator: IntValidator {bottom: 0; top: 120;}
                            height:30
                        }
                    }

                    onAccepted: {
                        console.log("Name:", nameField)
                        console.log("lastName:", lastNameField)
                        console.log("Age:", ageField)
                        console.log("NationalCode:", nationalCodeField)
                        fileHandler.savePersonInfo(nameField, lastNameField, parseInt(ageField), nationalCodeField)
                    }

                    onRejected: {
                        nameInputSix.text = ""
                        lastNameInputSix.text = ""
                        ageInputSix.text = ""
                        nationalCodeInputSix.text = ""
                    }
                }
            }

            Rectangle{
                // مدل داده‌ها

                id: donatChartSix
                width: parent.width / 3 + 100
                height: parent.height / 3.3
                color: "#424242"

                border.width: 1.5
                anchors{
                    top: linePlot.bottom
                    topMargin: 20
                    left: tableTimeEmotion.right
                    leftMargin: 20
                    //horizontalCenter: parent.horizontalCenter
                }
                ChartView {
                    id:donutParent
                    width: parent.width
                    height: parent.height
                    backgroundColor: "#424242"
                    legend.labelColor: "white" // رنگ نوشته‌های legend                        antialiasing: true

                    // نمودار دایره‌ای
                    PieSeries {
                        id: pieSeries
                        holeSize: 0.5  // اندازه حفره برای ایجاد شکل دونات

                        // احساسات مختلف و مقدارشان
                        PieSlice {id:angerSlice ; label: "Anger"; value: 0; color: "red" }
                        PieSlice {id:happinessSlice ; label: "Happiness"; value: 0; color: "green" }
                        PieSlice {id:fearSlice ; label: "Fear"; value: 0; color: "purple" }
                        PieSlice {id:sadnessSlice ; label: "Sadness"; value: 0; color: "blue" }
                        PieSlice {id:disgustSlice ; label: "Disgust"; value: 0; color: "yellow" }
                        PieSlice {id:surpriseSlice ; label: "Surprise"; value: 0; color: "pink" }
                        PieSlice {id:neutralSlice ; label: "Neutral"; value: 0; color: "orange" }
                    }
                    // تابع برای آپدیت مقدار احساسات در نمودار
                    function updateDonutChart(emotions) {
                        angerSlice.value = emotions["anger"]
                        happinessSlice.value = emotions["happiness"]
                        fearSlice.value    = emotions["fear"]
                        sadnessSlice.value  = emotions["sadness"]
                        disgustSlice.value  = emotions["disgust"]
                        surpriseSlice.value = emotions["surprise"]
                        neutralSlice.value  = emotions["neutral"]

                    }
                }


            }
            Connections {
                target: fileHandler
                function onEmotionRsult(emotion_status, emotion_prob, total_prob) {
                    emotionDataSix.emotions["anger"] = total_prob[0] * 100;
                    emotionDataSix.emotions["disgust"] = total_prob[1] * 100;
                    emotionDataSix.emotions["fear"] = total_prob[2] * 100;
                    emotionDataSix.emotions["happiness"] = total_prob[3] * 100;
                    emotionDataSix.emotions["neutral"] = total_prob[4] * 100;
                    emotionDataSix.emotions["sadness"] = total_prob[5] * 100;
                    emotionDataSix.emotions["surprise"] = total_prob[6] * 100;
                    percentageEmotion = emotion_prob;
                    emotionStatus = emotion_status;
                    chartView.addDataPoint(emotionDataSix.emotions);
                    newEmotions[emotion_status] += 1;
                    donutParent.updateDonutChart(newEmotions);
                    emotionDataSix.emotionsQmlChanged();
                }
            }
        }
    }

    FileDialog {
        id: videoDialog
        title: "Please choose a video file"
        nameFilters: ["Video files (*.mp4 *.avi *.mkv)"]
        onAccepted: {
            console.log("Selected video:", selectedFile)
            backEnd.setVideoPath(selectedFile)
        }
    }

    Connections {
        target: backEnd
        function onProcessingStatus(status, progress) {
            videoProgressBar.visible = true
            processingStatusText.visible = true
            processingStatusText.text = status
            videoProgressBar.value = progress
            
            if (progress >= 100 || progress <= 0) {
                // Hide progress bar after 2 seconds when processing is complete
                timer.start()
            }
        }
    }

    Timer {
        id: timer
        interval: 2000
        repeat: false
        onTriggered: {
            videoProgressBar.visible = false
            processingStatusText.visible = false
        }
    }

}

