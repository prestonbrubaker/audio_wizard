import os
from pydub import AudioSegment
import numpy as np

def normalize_audio(input_folder, output_folder, target_rms=-20):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Load audio
                audio = AudioSegment.from_mp3(input_path)
                
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples())
                
                # Calculate RMS
                rms = np.sqrt(np.mean(samples**2))
                
                # Check if RMS is valid
                if not np.isnan(rms) and rms != 0:
                    # Calculate scaling factor
                    scale_factor = 10**((target_rms - 20 * np.log10(rms)) / 20)
                    
                    # Apply normalization
                    normalized_samples = (samples * scale_factor).astype(np.int16)
                    
                    # Convert back to AudioSegment
                    normalized_audio = AudioSegment(normalized_samples.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=1)
                    
                    # Export normalized audio
                    normalized_audio.export(output_path, format="mp3")
                    print(f"Normalized {filename} and saved to {output_path}")
                else:
                    print(f"Skipping {filename} due to invalid RMS.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_folder = "copy"
output_folder = "normalizedampmp3"
target_rms = -20  # Target RMS level in dB
normalize_audio(input_folder, output_folder, target_rms)
