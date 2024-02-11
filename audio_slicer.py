from pydub import AudioSegment
import os

# Base directory where the subfolders are located
base_input_folder = 'fma_small'
output_folder = 'audiophiles'  # The single output directory for all sliced files

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def slice_mp3(file_path, output_dir, file_name):
    audio = AudioSegment.from_mp3(file_path)
    slice_duration = 10 * 1000  # 10 seconds in milliseconds
    for i in range(3):  # Each file is sliced into 3 parts
        start_ms = i * slice_duration
        end_ms = start_ms + slice_duration
        slice = audio[start_ms:end_ms]
        slice_name = f"{file_name}_slice_{i+1}.mp3"
        slice.export(os.path.join(output_dir, slice_name), format="mp3")
        print(f"Exported: {slice_name}")

# Iterate through each subfolder and process the MP3 files
for i in range(156):  # From 000 to 155
    subfolder = f"{i:03}"  # Format the folder number with leading zeros
    current_folder = os.path.join(base_input_folder, subfolder)
    print(f"Processing folder: {current_folder}")
    
    for file in os.listdir(current_folder):
        if file.endswith(".mp3"):
            file_path = os.path.join(current_folder, file)
            # Construct a unique file name using subfolder and file name
            unique_file_name = f"{subfolder}_{file[:-4]}"  # Exclude the '.mp3' extension
            slice_mp3(file_path, output_folder, unique_file_name)

print("Slicing completed.")
