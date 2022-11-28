# ECE460J Final Project
---
## Team Members
- Laith Altarabishi
- Sidharth Babu
- Afnan Mir
- Jonathan Vasilyev

---
## Project Overview
LoFi music is an essential element for college students in their studying, and in this project, we look to generate new LoFi music using deep learning.

We will be training an LSTM model on [this](https://github.com/nmtremblay/lofi-samples/tree/b1ca2ee1dc4606fb927f4db414bb612772beb479) dataset of MIDI files, and use this trained model to generate new MIDI files. MIDI files are music files that contain information about the notes, chords and instruments used in a song. We use this file format to train on because it provides easy to use sequential data that we can convert into vectors to train and generate music.