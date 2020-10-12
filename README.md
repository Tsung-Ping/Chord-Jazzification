# Chord-Jazzification
A deep learning approach to the chord coloring and chord voicing. A dataset featuring the interpretations of chord symbols is also propossed.

See the paper for more information: [Chord Jazzification: Learning Jazz Interpretations of Chord Symbols (ISMIR 2020)](https://program.ismir2020.net/static/final_papers/90.pdf)

## Requirements
* tensorflow-gpu 1.8.0 <br />
* numpy 1.16.2 <br />
* pretty_midi 0.2.9 <br />


## Descriptions
* `Chord_Jazzification_Dataset`: the annotations of 50 pop-jazz piano solos
* `chord_jazzification_preprocessing.py`: preprocess the chord jazzification dataset and get the preprocessed data `chord_jazzification_training_data.pickle`
   
   To listen to the chord progressions of the dataset, uncomment the following code: <br />
	 ```#generate_midi_instance(corpus['1'], 'example.mid', qpm=120, play_midi=True, show_pianoroll=False)```

   You can listen to other pieces of the dataset by changing the key in `corpus`; valid keys = {'1'-'50'}

* `Chord_Jazzification.py`: either to train the models or to inference chord sequences using the pre-trained models

   The pre-trained models are saved in the directories: `coloring_model`, `voicing_model` <br />
   The jazzifications of the JAAH dataset are saved in the directory: `JAAH_inference`

* `chord_jazzification_models.py`: the implementations of the models
