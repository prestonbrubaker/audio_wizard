from pydub import AudioSegment
import numpy as np
import os

# Define the directory containing your MP3 files
directory = 'copy'

# Path for the output text document
output_file = 'audio_diagnostics.txt'

# Function to compute RMS of an audio segment
def compute_rms(audio_segment):
    # Extract raw audio data as numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:  # Stereo
        left = samples[::2]
        right = samples[1::2]
        rms_value = np.sqrt(np.mean(np.square(left), dtype=np.float64) + np.mean(np.square(right), dtype=np.float64)) / 2
    else:  # Mono
        rms_value = np.sqrt(np.mean(np.square(samples), dtype=np.float64))
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
