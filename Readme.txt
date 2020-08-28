1. Run chord_jazzification_preprocessing.py to preprocess the chord jazzification dataset and get chord_jazzification_training_data.pickle
   
   To listen to the chord pregressions of the dataset, run the following code:
	 generate_midi_instance(corpus['1'], 'example.mid', qpm=120, play_midi=True, show_pianoroll=False)

   You can listen to any piece by change the key in corpus[key], where key = {'1'-'50'}