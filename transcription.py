import os
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
from openai import OpenAI
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, required=True,
                    help = "Audio file to be transcribed.")
parser.add_argument("--lang", type = str, required = True,
                    help = "Audio Language.")
args = parser.parse_args()

# Set OpenAI API key (replace 'your-api-key' with your actual API key)
with open("openai_key.txt", "r") as f:
    api_key = f.read()
OPENAI_CLIENT = OpenAI(api_key=api_key)

# Step 1: Load and convert the m4a file to wav using pydub
def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")
    return output_file

# Step 2: Reduce background noise using noisereduce
def reduce_noise(input_wav, output_folder):
    # Load audio data
    data, rate = sf.read(input_wav)
    print("Data shape (frames x channels):", data.shape)
    print("Sample rate:", rate)
    
    # Reduce noise
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease = 0.2, n_std_thresh_stationary = 2.5, n_fft = 1024)

    # Split in multiple files
    framestep = 10**7
    T = reduced_noise.shape[0]
    file_list = [reduced_noise[t:t+framestep] for t in range(0, T, framestep)]
    
    # Save the reduced noise audio to a new file
    for i,segment in enumerate(file_list):
        sf.write(os.path.join(output_folder, f"denoised_{i}.wav"), segment, rate)

    return output_folder

# Step 3: Transcribe speech using Whisper API
def transcribe_audio(file_path, language):
    print(f"\t{file_path}")
    audio_file = open(file_path, 'rb')
    transcript = OPENAI_CLIENT.audio.transcriptions.create(
        model = "whisper-1", 
        file = audio_file, 
        language = language
    )
    return transcript.text

# Main function
if __name__ == "__main__":

    if not os.path.exists("transcriptions"):
        os.makedirs("transcriptions")

    # File paths
    filename = args.file.split("/")[-1].split(".")[0]
    folder = args.file.rsplit("/", maxsplit = 1)[0]
    input_m4a = args.file # Input m4a file
    temp_wav = args.file.rsplit(".", maxsplit = 1)[0] + ".wav" # Temporary wav file
    noise_reduced_folder = args.file.rsplit(".", maxsplit = 1)[0] + "_denoised/"  # Final noise-reduced wav file

    if not os.path.exists(noise_reduced_folder):
        os.makedirs(noise_reduced_folder)

    # Step 1: Convert m4a to wav
    print("Converting m4a to wav...")
    if input_m4a.endswith(".m4a"):
        convert_m4a_to_wav(input_m4a, temp_wav)

    # Step 2: Reduce background noise
    print("Reducing background noise...")
    reduce_noise(temp_wav, noise_reduced_folder)

    # Step 3: Transcribe the cleaned audio using Whisper API
    print("Transcribing audio...")
    transcribed_segments = []
    for file in os.listdir(noise_reduced_folder):
        file_number = int(file.split(".")[0].split("_")[-1])
        transcription = transcribe_audio(os.path.join(folder, file), language=args.lang)
        transcribed_segments.append((file_number, transcription))

    # Output the transcription
    transcribed_segments = list(map(lambda tup: tup[-1], sorted(transcribed_segments, key = lambda tup: tup[0])))
    with open(os.path.join("transcriptions", filename + ".txt"), "w") as f:
        f.write("\n".join(transcribed_segments))

    # Cleanup: Remove temporary wav files if needed
    os.remove(temp_wav)
