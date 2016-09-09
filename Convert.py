from scipy.io import wavfile
import subprocess
import glob
import os

# Lists all the wav files
wav_files_list = glob.glob('/media/mateusz/Data/Magisterka/Datasets/TIMIT/Raw2/*/*/*.WAV')

# Create temporary names for the wav files to be converted. They will be renamed later on.
wav_prime = []
for f in wav_files_list:
    fileName, fileExtension = os.path.splitext(f)
    fileName += 'b'
    wav_prime.append(fileName + fileExtension)

# Command strings
cmd = "sox {0} -t wav {1}"
mv_cmd = "mv {0} {1}"

# Convert the wav_files first. Remove it. Rename the new file created by sox to its original name
for i, f in enumerate(wav_files_list):
    subprocess.call(cmd.format(f, wav_prime[i]), shell=True)
    os.remove(f)
    subprocess.call(mv_cmd.format(wav_prime[i], f), shell=True)