import psycopg2
from datetime import datetime

# PostgreSQL connection config — update with your credentials
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'animal_detector_db'
DB_USER = 'your_user'
DB_PASSWORD = 'your_password'

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def log_face_login(user, status):
    """Log a face login attempt with username, status (success/fail), and timestamp."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO face_logins (username, status, timestamp)
                    VALUES (%s, %s, %s)
                    """,
                    (user, status, datetime.utcnow())
                )
                conn.commit()
    except Exception as e:
        print(f"[DB ERROR - Face Login] {e}")

def log_detection(label, image_path):
    """Log detected label with image path and timestamp."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO detections (label, image_path, timestamp)
                    VALUES (%s, %s, %s)
                    """,
                    (label, image_path, datetime.utcnow())
                )
                conn.commit()
    except Exception as e:
        print(f"[DB ERROR - Detection] {e}")

def log_upload(local_path, drive_file_id):
    """Log upload event with local path, Google Drive file ID, and timestamp."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO uploads (local_path, drive_file_id, timestamp)
                    VALUES (%s, %s, %s)
                    """,
                    (local_path, drive_file_id, datetime.utcnow())
                )
                conn.commit()
    except Exception as e:
        print(f"[DB ERROR - Upload] {e}")
