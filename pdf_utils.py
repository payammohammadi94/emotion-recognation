from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF
import os

PDF_DIR_face_recognation = "pdf/faceRecognation"
PDF_DIR_fution_recognation = "pdf/futionRecognation"
PDF_DIR_voice_recognation = "pdf/voiceRecognation"
PDF_DIR_eeg_recognation = "pdf/eegRecognation"
class EmotionReport:
    def __init__(self):
        os.makedirs(PDF_DIR_face_recognation, exist_ok=True)
        os.makedirs(PDF_DIR_fution_recognation, exist_ok=True)
        os.makedirs(PDF_DIR_voice_recognation, exist_ok=True)
        os.makedirs(PDF_DIR_eeg_recognation, exist_ok=True)
        self.emotion_keys = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


    def generate_report_face_recognation(self, data_buffer):
        if not data_buffer:
            print("[-] No data to generate report.")
            return None

        all_emotions = [item['emotion'] for item in data_buffer]
        emotion_counter = Counter(all_emotions)
        total = len(all_emotions)
        average_prob = sum(item['prob'] for item in data_buffer) / total

        labels = list(emotion_counter.keys())
        sizes = list(emotion_counter.values())

        # Pie chart - بهبود شده با فونت بزرگتر
        plt.figure(figsize=(8, 8))
        colors = plt.cm.Set3(range(len(labels)))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                                          colors=colors, textprops={'fontsize': 10, 'fontweight': 'bold'})
        # بهبود فونت درصدها
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        plt.title("Face Emotion Distribution", fontsize=14, fontweight='bold', pad=20)
        pie_chart_path = "emotion_pie_chart.png"
        plt.savefig(pie_chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        timestamps = [item['timestamp'] for item in data_buffer]
        emotion_series = {key: [] for key in self.emotion_keys}

        for item in data_buffer:
            probs = item['probs']
            for i, key in enumerate(self.emotion_keys):
                emotion_series[key].append(probs[i])

        # رسم نمودار
        plt.figure(figsize=(12, 6))
        for key in self.emotion_keys:
            plt.plot(timestamps, emotion_series[key], marker='o', label=key)

        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.title("Emotion Trend Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Probability")
        plt.legend(loc='upper right')
        line_chart_path = "emotion_trend_chart.png"
        plt.savefig(line_chart_path)
        plt.close()

        # Bar chart - بهبود شده با خوانایی بیشتر
        plt.figure(figsize=(10, 6))
        # استفاده از رنگ‌های مختلف برای هر احساس
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        bars = plt.bar(range(len(labels)), sizes, color=colors_bar[:len(labels)], edgecolor='black', linewidth=1.5)
        
        # اضافه کردن مقدار روی هر میله
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=11, fontweight='bold')
        plt.yticks(fontsize=10)
        plt.title("Face Emotion Frequency", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("Emotion", fontsize=12, fontweight='bold')
        plt.ylabel("Count", fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        bar_chart_path = "emotion_bar_chart.png"
        plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Face Emotion Analysis Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Total Samples: {total}", ln=True)
        pdf.cell(0, 10, f"Average Confidence: {average_prob:.2f}%", ln=True)
        pdf.cell(0, 10, f"Most Frequent Emotion: {emotion_counter.most_common(1)[0][0]}", ln=True)
        pdf.ln(10)

        pdf.image(pie_chart_path, x=30, w=150)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Confidence Over Time", ln=True)
        pdf.image(line_chart_path, x=10, w=190)

        pdf.add_page()
        pdf.cell(0, 10, "Face Emotion Frequency (Bar Chart)", ln=True)
        pdf.image(bar_chart_path, x=30, w=150)
        # Save PDF
        report_filename = os.path.join(PDF_DIR_face_recognation, f"Emotion_Report_face_Recognation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(report_filename)

        # Remove temporary chart images
        os.remove(pie_chart_path)
        os.remove(line_chart_path)
        os.remove(bar_chart_path)

        print(f"[✓] Report saved as {report_filename}")
        return report_filename

    def generate_report_voice_recognation(self, data_buffer):
        if not data_buffer:
            print("[-] No data to generate report.")
            return None

        all_emotions = [item['emotion'] for item in data_buffer]
        emotion_counter = Counter(all_emotions)
        total = len(all_emotions)
        average_prob = sum(item['prob'] for item in data_buffer) / total

        labels = list(emotion_counter.keys())
        sizes = list(emotion_counter.values())

        # Pie chart - بهبود شده با فونت بزرگتر
        plt.figure(figsize=(8, 8))
        colors = plt.cm.Set3(range(len(labels)))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                                          colors=colors, textprops={'fontsize': 10, 'fontweight': 'bold'})
        # بهبود فونت درصدها
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        plt.title("Voice Emotion Distribution", fontsize=14, fontweight='bold', pad=20)
        pie_chart_path = "emotion_pie_chart.png"
        plt.savefig(pie_chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Bar chart - بهبود شده با خوانایی بیشتر
        plt.figure(figsize=(10, 6))
        # استفاده از رنگ‌های مختلف برای هر احساس
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        bars = plt.bar(range(len(labels)), sizes, color=colors_bar[:len(labels)], edgecolor='black', linewidth=1.5)
        
        # اضافه کردن مقدار روی هر میله
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=11, fontweight='bold')
        plt.yticks(fontsize=10)
        plt.title("Voice Emotion Frequency", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("Emotion", fontsize=12, fontweight='bold')
        plt.ylabel("Count", fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        bar_chart_path = "emotion_bar_chart.png"
        plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Voice Emotion Analysis Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Total Samples: {total}", ln=True)
        pdf.cell(0, 10, f"Average Confidence: {average_prob:.2f}%", ln=True)
        pdf.cell(0, 10, f"Most Frequent Emotion: {emotion_counter.most_common(1)[0][0]}", ln=True)
        pdf.ln(10)

        pdf.image(pie_chart_path, x=30, w=150)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Emotion Frequency (Bar Chart)", ln=True)
        pdf.image(bar_chart_path, x=10, w=190)
        # Save PDF
        report_filename = os.path.join(PDF_DIR_voice_recognation, f"Emotion_Report_voice_Recognation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(report_filename)

        # Remove temporary chart images
        os.remove(pie_chart_path)
        os.remove(bar_chart_path)

        print(f"[✓] Report saved as {report_filename}")
        return report_filename
    

    def generate_report_fution_recognation(self, data_buffer):
        if not data_buffer:
            print("[-] No data to generate report.")
            return None

        all_emotions = [item['emotion'] for item in data_buffer]
        emotion_counter = Counter(all_emotions)
        total = len(all_emotions)
        average_prob = sum(item['prob'] for item in data_buffer) / total

        labels = list(emotion_counter.keys())
        sizes = list(emotion_counter.values())

        # Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Fution Emotion Distribution")
        pie_chart_path = "emotion_pie_chart.png"
        plt.savefig(pie_chart_path)
        plt.close()

        # Line chart - confidence over time
        timestamps = [item['timestamp'] for item in data_buffer]
        probs = [item['prob'] for item in data_buffer]
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, probs, marker='o')
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.tight_layout()
        plt.title("Confidence Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Confidence (%)")
        line_chart_path = "confidence_line_chart.png"
        plt.savefig(line_chart_path)
        plt.close()

        # Bar chart - emotion counts
        plt.figure(figsize=(6, 4))
        plt.bar(labels, sizes, color='skyblue')
        plt.title("Fution Emotion Frequency")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        bar_chart_path = "emotion_bar_chart.png"
        plt.savefig(bar_chart_path)
        plt.close()

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Fution Emotion Analysis Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Total Samples: {total}", ln=True)
        pdf.cell(0, 10, f"Average Confidence: {average_prob:.2f}%", ln=True)
        pdf.cell(0, 10, f"Most Frequent Emotion: {emotion_counter.most_common(1)[0][0]}", ln=True)
        pdf.ln(10)

        pdf.image(pie_chart_path, x=30, w=150)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Fution Confidence Over Time", ln=True)
        pdf.image(line_chart_path, x=10, w=190)

        pdf.add_page()
        pdf.cell(0, 10, "Fution Emotion Frequency (Bar Chart)", ln=True)
        pdf.image(bar_chart_path, x=30, w=150)
        # Save PDF
        report_filename = os.path.join(PDF_DIR_fution_recognation, f"Emotion_Report_Fution_Recognation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(report_filename)

        # Remove temporary chart images
        os.remove(pie_chart_path)
        os.remove(bar_chart_path)

        print(f"[✓] Report saved as {report_filename}")
        return report_filename
    
    def generate_report_eeg_recognation(self, data_buffer):
        """تولید گزارش PDF برای EEG Emotion Recognition"""
        if not data_buffer:
            print("[-] No data to generate report.")
            return None

        all_emotions = [item['emotion'] for item in data_buffer]
        emotion_counter = Counter(all_emotions)
        total = len(all_emotions)
        average_prob = sum(item['prob'] for item in data_buffer) / total

        labels = list(emotion_counter.keys())
        sizes = list(emotion_counter.values())

        # Pie chart - بهبود شده با فونت بزرگتر
        plt.figure(figsize=(8, 8))
        colors = plt.cm.Set3(range(len(labels)))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                                          colors=colors, textprops={'fontsize': 10, 'fontweight': 'bold'})
        # بهبود فونت درصدها
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        plt.title("EEG Emotion Distribution", fontsize=14, fontweight='bold', pad=20)
        pie_chart_path = "emotion_pie_chart.png"
        plt.savefig(pie_chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Bar chart - بهبود شده با خوانایی بیشتر
        plt.figure(figsize=(10, 6))
        # استفاده از رنگ‌های مختلف برای هر احساس
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        bars = plt.bar(range(len(labels)), sizes, color=colors_bar[:len(labels)], edgecolor='black', linewidth=1.5)
        
        # اضافه کردن مقدار روی هر میله
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=11, fontweight='bold')
        plt.yticks(fontsize=10)
        plt.title("EEG Emotion Frequency", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("Emotion", fontsize=12, fontweight='bold')
        plt.ylabel("Count", fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        bar_chart_path = "emotion_bar_chart.png"
        plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "EEG Emotion Analysis Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Total Samples: {total}", ln=True)
        pdf.cell(0, 10, f"Average Confidence: {average_prob:.2f}%", ln=True)
        pdf.cell(0, 10, f"Most Frequent Emotion: {emotion_counter.most_common(1)[0][0]}", ln=True)
        pdf.ln(10)

        pdf.image(pie_chart_path, x=30, w=150)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Emotion Frequency (Bar Chart)", ln=True)
        pdf.image(bar_chart_path, x=10, w=190)
        # Save PDF
        report_filename = os.path.join(PDF_DIR_eeg_recognation, f"Emotion_Report_EEG_Recognation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(report_filename)

        # Remove temporary chart images
        os.remove(pie_chart_path)
        os.remove(bar_chart_path)

        print(f"[✓] Report saved as {report_filename}")
        return report_filename
