import sqlite3
import numpy as np


class FaceDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                event_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def register_face(self, embedding):
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        cursor = self.conn.execute(
            "INSERT INTO faces (embedding) VALUES (?)",
            (embedding.tobytes(),)
        )

        self.conn.commit()

        return cursor.lastrowid

    def get_faces(self):
        rows = self.conn.execute(
            "SELECT face_id, embedding FROM faces"
        ).fetchall()

        gallery = {}

        for face_id, blob in rows:
            gallery[face_id] = np.frombuffer(
                blob,
                dtype=np.float32
            )

        return gallery

    def log_event(self, face_id, event_type):
        self.conn.execute(
            """
            INSERT INTO events (face_id, event_type)
            VALUES (?, ?)
            """,
            (face_id, event_type)
        )

        self.conn.commit()

    def close(self):
        self.conn.close()