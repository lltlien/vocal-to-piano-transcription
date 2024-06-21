from music21 import note
import numpy as np

# input: notes (pitch, start, end)
# output: save musicxml file

# add rests to the notes (pitch, start, end)
def add_rests(notes,bpm):
    new_notes = []
    new_intervals = []
    prev_end = 0
    for note, start, end in notes:
        if start - prev_end >= (60 / bpm) / 0.02 :
            new_notes.append(0)
            new_intervals.append((prev_end, start))
        new_notes.append(note)
        new_intervals.append((start, end))
        prev_end = end
    print('new_notes:', new_notes)
    print('new_intervals:', new_intervals)
    return new_notes, new_intervals
    

def note_length_beats(start, end):
    note_length_beats = (end - start) / 4
    supported_values_note = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
    note_length = min(supported_values_note, key=lambda x: abs(x - note_length_beats))

    return note_length

def rest_length_beats(start, end):
    rest_length_beats = (end - start) / 4
    supported_values_rest = [0, 0.5, 1, 2, 3, 4]
    rest_length = min(supported_values_rest, key=lambda x: abs(x - rest_length_beats))

    return rest_length

def create_music_xml(notes, start_beat_time, output_path, bpm, beat_per_bar):
    notes, intervals = add_rests(notes, bpm)

    score = stream.Score()
    part = stream.Part()
    # add bpm 
    part.append(tempo.MetronomeMark(number=bpm))
    part.append(meter.TimeSignature(f'{beat_per_bar}/4'))

    midi_notes = [note for note in notes]

    temp_stream = stream.Stream()
    for midi_note in midi_notes:
        n = note.Note(midi_note)
        temp_stream.append(n)
    ks = temp_stream.analyze('key')
    part.append(key.KeySignature(ks.sharps))

    start_music = start_beat_time * 4 / (60 / bpm)
    fist_onset = intervals[0][0]
    for midi_note, (start_time, end_time) in zip(notes, intervals):
        if start_time < start_music:
            continue
    
        tertiary_per_bar = 4 * beat_per_bar

        start_bar_index = int((start_time-fist_onset) // tertiary_per_bar) + 1
        end_bar_index = int(end_time // tertiary_per_bar) + 1

        if midi_note == 0: 
            n1 = note.Rest()
            n1.quarterLength = rest_length_beats(start_time, min(end_time, (start_bar_index) * tertiary_per_bar))
        else:
            n1 = note.Note()
            n1.pitch.midi = midi_note 
            n1.quarterLength = note_length_beats(start_time, min(end_time, (start_bar_index) * tertiary_per_bar))

        if n1.quarterLength == 0:
            continue

        while len(part.getElementsByClass(stream.Measure)) < start_bar_index:
            part.append(stream.Measure(number=len(part.getElementsByClass(stream.Measure)) + 1))

        part.measure(start_bar_index).append(n1)

        if start_bar_index != end_bar_index:
            n1.tie = tie.Tie('start')

            if midi_note != 0:
                n2 = note.Note()
                n2.pitch.midi = midi_note 
                n2.quarterLength = note_length_beats((start_bar_index) * tertiary_per_bar, end_time)
                n2.tie = tie.Tie('stop')

            if midi_note == 0:
                n2 = note.Rest()
                n2.quarterLength = rest_length_beats((start_bar_index) * tertiary_per_bar, end_time)

            if n2.quarterLength == 0:
                continue

            while len(part.getElementsByClass(stream.Measure)) < end_bar_index:
                part.append(stream.Measure(number=len(part.getElementsByClass(stream.Measure)) + 1))

            part.measure(end_bar_index).append(n2)
            print(n1.quarterLength, n2.quarterLength)

    score.append(part)
    score.write('musicxml', fp=output_path)
