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

Once we create our long sequence of notes, we prepare the pairs of input sequences and output notes/chords. First we define a sequence length of $N$. In our case, we chose 50. Then, each unique note/chord gets assigned an index that we can refer to.

```python
note_dict = dict((note, number) for number, note in enumerate(pitch_names))
```
We use this dictionary to create our input sequences and our output. To create our pairs, we iterate through our sequence of notes. At each iteration, we take the next $N$ notes, and use there indices to create our input sequence. This gives us an $N$-dimensional vector we can use as our input. We then take the $N + 1$th note and use its index to create our output note/chord. However, with our current representation, we would be predicting a scalar number for each input sequence. We would like to predict a vector of probabilities of every possible note/chord. To do this, we represent our output indices as one-hot encoded vectors.

```python
from keras.utils.np_utils import to_categorical
network_output = to_categorical(network_output)
```
After iterating through the whole sequence, we produce all of our input sequences and output note/chord pairs.

## The Model
For our model, we chose to use a sequence-to-sequence model, more specifically an Long Short-Term Memory (LSTM) model. This is a type of recurrent neural network that is able to retain information from long term dependencies far better than normal recurrent neural networks.

Our model looks like the following:
```python
model = Sequential()

    model.add(LSTM(
        units=512,
        input_shape = (input.shape[1], input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))

    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))

    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))

    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```
We opt for 2 LSTM layers with 512 units each, and a final LSTM layer with 256 units. We then add one fully connected layer with a dropout layer. Finally, we add the output layer with a softmax operation to get our output probabilities. We then train this model for 100 epochs with a batch size of 64.

## Generating Music
Now that we have trained the model, its time to generate some music! We start by choosing a random note to start with. We then iterate 500 times, each time predicting the next note/chord, taking the note/chord with the highest probability and adding it to our output sequence. For each iteration, we use the information from the previous $N$ notes to predict the next note/chord. Additionally, when we create our full output sequence, each note/chord is required an offset value. This is the amount of time that should pass before the next note/chord is played. For all notes, we assign the offset values at 0.5 intervals, as this was the most common interval found when investigating the MIDI files. 

Here is the final MIDI file that we generated:

[insert MIDI file here]

Note that this is a file containing only notes and chords with no instruments and a far too fast tempo. With the help of some music editing magic, adding a beat and instruments, we create the following song:

[insert edited song here]

This begins to sound more and more like a song, although there are still some issues.

## Further Exploration
After getting this baseline, we attempted to explore a little more. Our exploration included trying a variational autoencoder (VAE) model to generate music. Autoencoders are pairs of two neural networks, an encoder and a decoder model, and it is the job of the encoder to find a way to compress a given input dataset into the latent space - so that the reconstructed output we get later with the decoder is similar to the input. The variational subset of autoencoders have very nice properties that allow for us to reconstruct music from points in the latent space. In contrast to our previous methodology, we try to use a generative model approach rather than a seq2seq model approach to generate our music and compare the results in the end.

Specifically, we used MusicVAE, an open-source, hierarchical recurrent VAE that is provided by Magenta for the underlying model of our code, and trained on the same dataset of lofi music as before. Similarly to our data pre-processing before, we take all of our MIDI files from our data set, and convert them into sequences of notes that can be fed into our model. Once we have our converted sequences, we can then choose a specific music configuration for our network to train on. These configurations ranged from different 2-bar to 16-bar melodies, as well as different instruments being introduced, such as bass and drums. For the purposes of our minimalistic music setting, we decided to configure the network with a simple 2-bar melody configuration and trained it on our lofi dataset.

Using this method, we were able to create the following MIDI file:

[insert MIDI file here]

and after some editing, we were able to create this final song:

[inser edited song here]

This again, sounds somewhat like an actual song, but it seems we run into some of the same issues as before. There are noticeably more complex rhythms than before, but at the cost of inconsistency in the global 'smoothness' of the song - as many measures seem to have very sporadic rhythm. Ultimately, we generated this music by sampling the model a certain amount of times, and perhaps we would have more successful results if we were able to condition the network based on the samples we've drawn thus far (a historical conditioning of sorts). 

## Conclusion and Reflection

In this project we hoped to generate LoFi music using deep learning techniques. We used models such as LSTM's and VAE's to attempt to meet this goal, and while we were able to generate some music, we ran into a couple main issues. The first main issue was the fact that the models were not able to learn the notion of a key sign for music. The models could not figure out that certain sharps, flats, and chords would only work in certain circumstances. Although audio imperfections are an aspect of LoFi music, we would still like to have some satisfying progressions. The second main issue, which may have caused the first issue, could have been the lack of large amounts of data. We were only able to compile a little over 100 MIDI files, each containing small amounts of notes in ther. Especially for training a deep learning model from scratch, this may not have been enough data to learn all the intricacies of the music. In the future, to create better performance, we would like to collect more data to train on, and also attempt to use more modern techniques, which include finetuning large pre-trained generative models to perform our task.

If you would like to check out our code, it can be found [here]!