# Using Deep Learning Techniques to Generate LoFi Music
 
## Authors
Laith Altarabishi, Sidharth Babu, Afnan Mir, Jonathan Vasilyev

## Introduction
Low Fidelity music, or more famously known as LoFi music, is an important part of almost every college student's life. It is a genre of music that is characterized by its low quality audio, environmental sounds, and audio imperfections. It is often used as background music for studying, working, and relaxing. In this project, we look to use deep learning techniques to generate new LoFi music.

We looked to create a sequence-to-sequence deep learning model that would take in a sequence of notes and chords and output a new sequence of notes and chords. Some challenges we faced in this project was a lack of a large dataset of LoFi music, and vectorizing the sequence of notes and chords into a format that would be usable by a model.

## The Dataset
The dataset we used for this project can be found [here](https://github.com/nmtremblay/lofi-samples). It contains over 100 MIDI files for training. MIDI files, or Musical Instrument Digital Interface files, are a type of file that contain information about the notes, chords, and instruments that are used in a song. They are far smaller than typical audio files (.wav and .mp3) because MIDI files do not hold any information about the actual audio.

We chose to use MIDI files as our training dataset because they provide easy-to-use sequential data that we can convert into vectors without too much trouble. This will allow us to create our training dataset quite trivially.

This dataset contains numerous samples of LoFi music, each containing a sequence of about 4-5 notes and chords.

## Data Preprocessing
In order to use the MIDI files as our training dataset, we had to convert our data into a format that would be usable by our model. Our model should be able to take in an input sequence of notes and/or chords and output 1 new note or chord. Our first step was to take all of our MIDI files and concatenate them into one large sequence of notes and chords to allow us to more easily create input sequences and output note/chord pairs.

```python
import glob
import pickle
from music21 import converter, instrument, note, chord
notes = []
for file in glob.glob("../lofi-samples/samples/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = None
    
    try:
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes
    
    for parsed_note in notes_to_parse:
        if isinstance(parsed_note, note.Note):
            notes.append(str(parsed_note.pitch))
        elif isinstance(parsed_note, chord.Chord):
            notes.append('.'.join(str(n) for n in parsed_note.normalOrder))
with(open("../data/notes", "wb")) as filepath:
    pickle.dump(notes, filepath)
```

We represent each note as a string of the pitch of the note, and each chord as a string of the pitches of the notes in the chord separated by a period. This gives each note and each chord a unique string representation that we can use to create our input sequences and output note/chord pairs.

