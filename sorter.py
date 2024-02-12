import os
from pydub import AudioSegment

# Function to check if a file is an MP3 and its duration is not 10 seconds
def should_delete(file_path):
    if file_path.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
        return len(audio) != 10000  # Duration is in milliseconds (10 seconds = 10000 milliseconds)
    return False

# Function to delete files that are not 10 seconds in length
def delete_files(folder_path):
    deleted_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if should_delete(file_path):
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted: {file_path}")
    print(f"Total files deleted: {deleted_count}")

# Replace 'folder_path' with the path to your folder containing the MP3 files
folder_path = 'copy'
delete_files(folder_path)
