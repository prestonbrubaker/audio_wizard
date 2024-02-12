import os
from pydub import AudioSegment
from pydub.utils import mediainfo

def calculate_rms(mp3_file_path):
    """Calculate RMS of an MP3 file."""
    audio = AudioSegment.from_mp3(mp3_file_path)
    return audio.rms

def process_folder(folder_path, output_file):
    """Process all MP3 files in a folder, recording their RMS values to a text document."""
    with open(output_file, 'w') as outfile:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mp3"):
                    full_path = os.path.join(root, file)
                    try:
                        rms_value = calculate_rms(full_path)
                        outfile.write(f"{file}: {rms_value}\n")
                        print(f"Processed {file}")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

# Usage
folder_path = "copy"  # Replace with your folder path
output_file = "rms_values.txt"
process_folder(folder_path, output_file)
