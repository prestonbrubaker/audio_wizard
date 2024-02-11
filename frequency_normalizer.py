from pydub import AudioSegment
import glob
import os

# Set the path to your MP3 files folder
folder_path = 'flat_data'

# Iterate over all MP3 files in the folder
for mp3_file in glob.glob(os.path.join(folder_path, '*.mp3')):
    audio = AudioSegment.from_file(mp3_file)
    
    # Check if the sample rate is 48000 Hz
    if audio.frame_rate == 48000:
        print(f"Converting {mp3_file} from 48000 Hz to 44100 Hz")
        # Convert the sample rate to 44100 Hz
        converted_audio = audio.set_frame_rate(44100)
        
        # Save the converted file, overwrite the original or save with a new name as needed
        converted_audio.export(mp3_file, format="mp3")  # Overwrites the original file
        # For saving with a new name, modify the export line accordingly
        # converted_audio.export(mp3_file.replace('.mp3', '_converted.mp3'), format="mp3")
