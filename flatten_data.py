import os
from pydub import AudioSegment

# Define the directory containing your MP3 snippets
source_directory = 'raw_data'
# Define the directory to save the converted mono files
mono_directory = os.path.join(source_directory, '../flattened_data')

# Create the mono directory if it doesn't exist
os.makedirs(mono_directory, exist_ok=True)

# Function to convert stereo to mono and save as a new file
def convert_stereo_to_mono_and_save(file_path, file_name):
    audio = AudioSegment.from_file(file_path)
    if audio.channels == 2:
        mono_audio = audio.set_channels(1)
        # Define the path for the new mono file
        new_file_path = os.path.join(mono_directory, file_name)
        mono_audio.export(new_file_path, format="mp3")

# Iterate over all MP3 files in the source directory
for file_name in os.listdir(source_directory):
    if file_name.endswith('.mp3'):
        file_path = os.path.join(source_directory, file_name)
        convert_stereo_to_mono_and_save(file_path, file_name)

print("Conversion and copying complete.")
