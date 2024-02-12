import os
from pydub import AudioSegment
import pyloudnorm as pyln

def normalize_audio(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Load audio
                audio = AudioSegment.from_mp3(input_path)
                
                # Normalize audio to -14 LUFS and -1.0 dBTP
                loudness = pyln.Loudness(unit='LUFS')
                normalized_audio = loudness.normalize(audio)
                
                # Export normalized audio
                normalized_audio.export(output_path, format="mp3")
                print(f"Normalized {filename} and saved to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_folder = "copy"
output_folder = "amplitudeconcerns"
normalize_audio(input_folder, output_folder)
