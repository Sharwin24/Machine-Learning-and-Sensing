import hashlib
import random
import os


def generate_dataset_id(name):
    """
    Generate a 5-digit random number based on the hash of the name.
    Use this as your dataset folder name.
    """
    hash_object = hashlib.md5(name.encode())
    hex_dig = hash_object.hexdigest()
    seed = int(hex_dig, 16) % (10**8)
    random.seed(seed)
    return random.randint(10000, 99999)


# Enter your name to generate your dataset ID
dataset_id = generate_dataset_id(name="Sharwin Patil")

# If the folder exists convert the audio files
if os.path.exists(f"{dataset_id}/"):
    audio_files = os.listdir(f"{dataset_id}")
    # Convert m4a to wav
    for file in audio_files:
        if file.endswith(".m4a"):
            os.system(
                f"ffmpeg -i {dataset_id}/{file} {dataset_id}/{file[:-4]}.wav"
            )
            print(f"Converted {file} to {file[:-4]}.wav")
            # Delete the original m4a file
            os.remove(f"{dataset_id}/{file}")
    print(f"Converted {len(audio_files)} files")
else:
    print(f"Dataset ID {dataset_id} does not exist.")
