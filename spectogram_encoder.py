import os

def create_combined_mel_spectrogram_image(file_path, output_folder):
    # Your existing code to create and save the spectrogram image
    pass

def process_audio_files(folder_path, output_folder):
    # Check if the output directory exists, create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file in os.listdir(folder_path):
        if file.endswith('.mp3'):
            file_path = os.path.join(folder_path, file)
            print(f"Processing file: {file}")
            create_combined_mel_spectrogram_image(file_path, output_folder)

# Example usage
input_folder = "copy"
output_folder = "copyspecto"
process_audio_files(input_folder, output_folder)
