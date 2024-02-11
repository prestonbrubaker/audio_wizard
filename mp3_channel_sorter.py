import os
from pydub import AudioSegment
from shutil import move

# Define the directory containing your MP3 snippets
source_directory = 'raw_data'
mono_directory = os.path.join(source_directory, 'mono')
stereo_directory = os.path.join(source_directory, 'stereo')

# Create the subdirectories if they don't exist
os.makedirs(mono_directory, exist_ok=True)
os.makedirs(stereo_directory, exist_ok=True)

# Function to check and move files
def sort_files_into_folders(file_path, file_name):
    audio = AudioSegment.from_file(file_path)
    if audio.channels == 1:
        destination = mono_directory
    elif audio.channels == 2:
        destination = stereo_directory
    else:
        # If the file is neither mono nor stereo, you might want to handle it differently
        return
    move(file_path, os.path.join(destination, file_name))

# Iterate over all MP3 files in the source directory
for file_name in os.listdir(source_directory):
    if file_name.endswith('.mp3'):
        file_path = os.path.join(source_directory, file_name)
        sort_files_into_folders(file_path, file_name)

print("Sorting complete.")
