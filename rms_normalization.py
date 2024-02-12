from pydub import AudioSegment
import numpy as np
import os

# Define the directory containing your MP3 files
directory = 'copy'

# Path for the output text document
output_file = 'audio_diagnostics.txt'

# Function to compute RMS of an audio segment, adjusted for potential data issues
def compute_rms(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)  # Ensure samples are float64
    if audio_segment.channels == 2:  # Stereo
        left = samples[::2]
        right = samples[1::2]
        # Ensure no invalid values; replace NaNs with 0 for calculation
        left = np.nan_to_num(left, nan=0.0)
        right = np.nan_to_num(right, nan=0.0)
        rms_value = np.sqrt(np.mean(np.square(left)) + np.mean(np.square(right))) / 2
    else:  # Mono
        samples = np.nan_to_num(samples, nan=0.0)  # Replace NaNs with 0
        rms_value = np.sqrt(np.mean(np.square(samples)))
    return rms_value

# Open or create the output file
with open(output_file, 'w') as file:
    # Write the header of the document
    file.write('Filename, RMS, Duration(s), Peak Amplitude, Format\n')
    
    # Process each MP3 file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            filepath = os.path.join(directory, filename)
            try:
                # Load the MP3 file
                audio = AudioSegment.from_mp3(filepath)
                
                # Compute diagnostics
                rms = compute_rms(audio)
                duration = len(audio) / 1000.0  # Convert from ms to s
                peak_amplitude = audio.max
                
                # Write diagnostics to file
                file.write(f'{filename}, {rms:.2f}, {duration}, {peak_amplitude}, MP3\n')
            except Exception as e:
                print(f'Error processing {filename}: {e}')

print('Audio diagnostics written to:', output_file)
