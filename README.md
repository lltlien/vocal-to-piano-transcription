# vocal-to-piano-transcription
Transcribe vocal melodies in songs to piano sheet music.
## Setup
### System
- **OS:** LINUX

### Clone this repository
```bash
git clone git@github.com:lltlien/vocal-to-piano-transcription.git
cd vocal-to-piano-transcription
```

### Install dependencies
```bash
pip install -r 'requirements.txt'
```

- Download model to `checkpoints` folder: [This link](https://github.com/lltlien/vocal-to-piano-transcription/releases/tag/latest)

## Usage
1. Open `getvocals.py` file and change link to your audio file, run following command to get vocal from music song (store in `vocals` folder):
```bash
$ python3 getvocals.py
```

2. Run this command to get file `.mid`:
```bash
$ python3 infer.py "your-vocal-audio-path"
```
3. Download [Muse Score](https://musescore.org/vi/download) to convert `.mid' to music sheet.
   
## References :

- *A Phoneme-Informed Neural Network Model for Note-Level Singing Transcription: [paper](https://arxiv.org/abs/2304.05917)*
- *Pseudo-Label Transfer from Frame-Level to Note-Level in a Teacher-Student Framework for Singing Transcription from Polyphonic Music: [paper](https://ieeexplore.ieee.org/document/9747147)*
