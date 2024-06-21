import argparse
import numpy as np
import torch
import librosa
import torchaudio


from phn_ast.midi import save_midi
from phn_ast.decoding import FramewiseDecoder
from phn_ast.model import TranscriptionModel
from phn_ast.savexml16 import *

def infer(model_file, input_file, output_file, pitch_sum, bpm, device, beat_per_bar):
    ckpt = torch.load('checkpoints/model.pt', map_location='cpu')
    config = ckpt['config']
    model_state_dict = ckpt['model_state_dict']
    ckpt['config']['onset_threshold'] = 0.2
    ckpt['config']['offset_threshold'] = 0.2
    print(ckpt['config'])

    model = TranscriptionModel(config)
    model.load_state_dict(model_state_dict)
    model.to(device)

    model.to(device)
    model.eval()

    model.pitch_sum = pitch_sum

    decoder = FramewiseDecoder(config)

    audio, sr = torchaudio.load(input_file)
    audio = audio.numpy().T
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio_re = librosa.resample(audio, orig_sr=sr, target_sr=config['sample_rate'])
    audio_re = torch.from_numpy(audio_re).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(audio_re)
        p, i = decoder.decode(pred, input_file)

    intervals = (np.array(i) * 0.02).reshape(-1, 2)
    start_beat_time = 0
    print('start_beat_time:', start_beat_time)

    p = np.array([round(midi) for midi in p])
    
    output_xml ='output/out.xml'
    print('time', np.array(i) * 0.02)
    ins = np.round(np.array(i) * 0.02 / (60/bpm) * 4)

    combined_data = list(zip(p, ins[:,0], ins[:,1]))
    combined_data = [(note, start, end) for note, start, end in combined_data if start != end]
    print('combined_data:', combined_data)

    create_music_xml(combined_data, start_beat_time ,output_xml,  bpm, beat_per_bar)

    save_midi(output_file, p, intervals, bpm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_file = 'checkpoints/model.pt'
    parser.add_argument('input_file', type=str)
    output_file = 'output/out.mid'
    parser.add_argument('--pitch_sum', default='weighted_mean', type=str)
    parser.add_argument('--bpm', '-b', default=120.0, type=float)
    parser.add_argument('--device', '-d',
                        default='cpu')

    args = parser.parse_args()

    infer(model_file, args.input_file,output_file, args.pitch_sum, args.bpm, args.device, beat_per_bar=3)