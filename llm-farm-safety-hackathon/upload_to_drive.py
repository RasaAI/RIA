from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Path to your service account json key file
SERVICE_ACCOUNT_FILE = 'service_account_key.json'

# Your Google Drive folder ID where you want to upload files
FOLDER_ID = '1mYDoEb5LMO66ogc0ywGI9ECBX5lOJQD8'

SCOPES = ['https://www.googleapis.com/auth/drive.file']

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=credentials)

def upload_file(filepath):
    file_metadata = {
        'name': filepath.split('\\')[-1],  # Extract filename from path, works on Windows
        'parents': [FOLDER_ID]
    }
    media = MediaFileUpload(filepath, resumable=True)
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    print(f"[Drive] Uploaded file ID: {file.get('id')}")
