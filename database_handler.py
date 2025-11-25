import sqlite3
import json
from datetime import datetime

class EmotionDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                emotion TEXT,
                prob REAL,
                probs TEXT,
                face_id INTEGER
            )
        ''')
        # Create index for face_id if it doesn't exist
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_id ON emotions(face_id)')
        except:
            pass
        conn.commit()
        conn.close()

    def save_batch(self, data_buffer):
        if not data_buffer:
            return

        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        try:
            for item in data_buffer:
                face_id = item.get("face_id", None)
                # تبدیل probs به لیست اگر numpy array است
                probs = item.get("probs", [])
                if hasattr(probs, 'tolist'):  # numpy array
                    probs = probs.tolist()
                elif not isinstance(probs, list):
                    probs = list(probs)
                
                cursor.execute('''
                    INSERT INTO emotions (timestamp, emotion, prob, probs, face_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    item["timestamp"],
                    item["emotion"],
                    item["prob"],
                    json.dumps(probs),
                    face_id
                ))
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"[-] Database error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def fetch_all(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, emotion, prob, probs, face_id FROM emotions")
        rows = cursor.fetchall()
        conn.close()

        data = []
        for row in rows:
            data.append({
                "timestamp": row[0],
                "emotion": row[1],
                "prob": row[2],
                "probs": json.loads(row[3]) if row[3] else [],
                "face_id": row[4] if len(row) > 4 else None
            })
        return data
    
    def fetch_by_face_id(self, face_id):
        """دریافت داده‌های یک صورت خاص"""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, emotion, prob, probs, face_id FROM emotions WHERE face_id = ?", (face_id,))
        rows = cursor.fetchall()
        conn.close()

        data = []
        for row in rows:
            data.append({
                "timestamp": row[0],
                "emotion": row[1],
                "prob": row[2],
                "probs": json.loads(row[3]) if row[3] else [],
                "face_id": row[4] if len(row) > 4 else None
            })
        return data
