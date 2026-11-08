import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "/tmp/visionguard.db" if os.getenv("VERCEL") else "instance/visionguard.db"))


def get_connection():
    if DATABASE_URL:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    if DATABASE_URL:
        for attempt in range(30):
            try:
                connection = get_connection()
                with connection:
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS prediction_history (
                                id SERIAL PRIMARY KEY,
                                filename TEXT NOT NULL,
                                prediction TEXT NOT NULL,
                                real_probability DOUBLE PRECISION NOT NULL,
                                ai_probability DOUBLE PRECISION NOT NULL,
                                model_version TEXT NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL
                            )
                        """)
                connection.close()
                return
            except Exception:
                if attempt == 29:
                    raise
                time.sleep(2)
    else:
        connection = get_connection()
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    real_probability REAL NOT NULL,
                    ai_probability REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        connection.close()


def save_prediction(filename, result):
    values = (
        filename,
        result["prediction"],
        result["probability_real"],
        result["probability_ai"],
        str(result["model_version"]),
        datetime.now(timezone.utc).isoformat()
    )

    connection = get_connection()
    try:
        if DATABASE_URL:
            with connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO prediction_history
                    (filename, prediction, real_probability, ai_probability,
                     model_version, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, values)
        else:
            connection.execute("""
                INSERT INTO prediction_history
                (filename, prediction, real_probability, ai_probability,
                 model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, values)
        connection.commit()
    finally:
        connection.close()


def get_history():
    connection = get_connection()
    try:
        if DATABASE_URL:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM prediction_history ORDER BY id DESC
                """)
                rows = cursor.fetchall()
        else:
            rows = connection.execute("""
                SELECT * FROM prediction_history ORDER BY id DESC
            """).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()