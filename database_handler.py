import sqlite3
import json
from datetime import datetime

class EmotionDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                emotion TEXT,
                prob REAL,
                probs TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_batch(self, data_buffer):
        if not data_buffer:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for item in data_buffer:
            cursor.execute('''
                INSERT INTO emotions (timestamp, emotion, prob, probs)
                VALUES (?, ?, ?, ?)
            ''', (
                item["timestamp"],
                item["emotion"],
                item["prob"],
                json.dumps(item["probs"])
            ))
        conn.commit()
        conn.close()

    def fetch_all(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, emotion, prob, probs FROM emotions")
        rows = cursor.fetchall()
        conn.close()

        data = []
        for row in rows:
            data.append({
                "timestamp": row[0],
                "emotion": row[1],
                "prob": row[2],
                "probs": json.loads(row[3])
            })
        return data
