from spleeter.separator import Separator
import os

def separate_vocals(input_file, output_dir):
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(input_file, output_dir)
    
    # Construct the path to the vocal file
    file_name = os.path.basename(input_file)
    name, ext = os.path.splitext(file_name)
    vocal_file = os.path.join(output_dir, name, 'vocals.wav')
    
    return vocal_file

input_file = '/home/lltlien/Downloads/icassp2022-vocal-transcription-main/phonemetrans/vocals/river.mp3'
output_dir = '/home/lltlien/Downloads/icassp2022-vocal-transcription-main/phonemetrans/vocals'

vocal_file_path = separate_vocals(input_file, output_dir)
print(f"Done! Vocals are saved in {vocal_file_path}")
