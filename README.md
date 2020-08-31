# Chord-Jazzification
A dataset for chord coloring and voicing.

## Requirements
   tensorflow-gpu 1.8.0
   numpy 1.16.2
   pretty_midi 0.2.9


## Descrpitions
1. Run `chord_jazzification_preprocessing.py` to preprocess the chord jazzification dataset and get `chord_jazzification_training_data.pickle`
   
   To listen to the chord progressions of the dataset, run the following code:
	 ```generate_midi_instance(corpus['1'], 'example.mid', qpm=120, play_midi=True, show_pianoroll=False)```

   You can listen to any piece in the dataset by change the key in corpus[key]; valid keys = {'1'-'50'}

2. Run `Chord_Jazzification.py` either to train the models or to inference chord sequences using pre-trained models

   The pre-trained models are saved in the directories: coloring_model, voicing_model
   The jazzifications of the JAAH dataset are saved in the directory: JAAH_inference
