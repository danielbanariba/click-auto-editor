from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from datetime import datetime, timedelta

def extract_info_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if not lines:
            return None, None
        title = lines[0].strip()
        description = '\n'.join(lines[1:])
    return title, description

def upload_video(youtube, file, info_file):
    title, description = extract_info_from_file(info_file)

    publish_at = (datetime.utcnow() + timedelta(days=1)).isoformat('T') + 'Z'
    
    body=dict(
        snippet=dict(
            title=title,
            description=description,
            tags="Python, YouTube API, Video Upload",
            categoryId="11",
        ),
        status=dict(
            privacyStatus='private',
            publishAt=publish_at  # Add this line
        ),
    )

    media = MediaFileUpload(file, mimetype='video/mp4')

    request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )

    response = request.execute()

    print(f'Video uploaded, ID: {response["id"]}')