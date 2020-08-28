import numpy as np
import itertools
from collections import Counter
import pickle
import xlrd
from fractions import Fraction
import pretty_midi as pm
import re
import matplotlib.pyplot as plt
import music21

def rate_keys(List, sorted_by_rate=False):
    c = Counter(List)
    if sorted_by_rate:
        return sorted([(i, c[i], round(c[i] / sum(c.values()), 4)) for i in c], key=lambda x: -x[2])
    else:
        return sorted([(i, c[i], round(c[i] / sum(c.values()), 4)) for i in c])

def note_name_to_number(note_name):
    '''convert note name to midi number'''

    def modify_number_by_accidental(accidental):
        m_table = {'#': 1, 'b': -1, 'x': 2, 'd': -2, '': 0}
        return m_table[accidental]

    note_name_without_accidental = re.sub('#|b|x|d', '', note_name)
    accidental = note_name[1] if note_name != note_name_without_accidental else ''

    try:
        note_number = pretty_midi.note_name_to_number(note_name_without_accidental) + modify_number_by_accidental(accidental)
    except:
        print('Error: invalid note name', note_name)
        exit(1)

    return note_number


def read_dataset(data_dir):
    quality_conversion_dict = \
        {'7': 'M', '7(+5)': 'a', '7(+9)': 'M', '7(-9)': 'M', '7(-9,13)': 'M', '7(13)': 'M', '7(9)': 'M',
         '7(9,+11)': 'M', '7(-9,+11)': 'M', '7(9,13)': 'M', '7(+9,-13)': 'M', '7(-13)': 'M', '7(9,-13)': 'M',
         '7(+11)': 'M', '7sus4': 'M',
         'M': 'M', 'M(2)': 'M', 'M6': 'M', 'M6(2)': 'M', 'M6(9)': 'M', 'M7': 'M', 'M7(13)': 'M', 'M7(+11)': 'M',
         'M7(9)': 'M', 'M7(9,13)': 'M', 'M7(11)': 'M', 'Madd2': 'M', 'sus4': 'M',
         'm': 'm', 'm6': 'm', 'm7': 'm', 'm7(11)': 'm', 'm7(9)': 'm', 'm7(9,11)': 'm', 'm7-5': 'm', 'm7-5(9)': 'm',
         'mM7': 'm', 'mM7(9)': 'm', 'madd2': 'm',
         'aug': 'a', 'aug7': 'a',
         'dim': 'd', 'dim7': 'd',
         'None': 'None'}
    def split_symbol(symbol):
        '''split chord symbol into root, quality, bass, and triad type'''
        if symbol != 'None':
            try:
                root = symbol.split(':')[0]
                squality = symbol.split(':')[1] if '/' not in symbol else (symbol.split('/')[0]).split(':')[1]
                bass = root if '/' not in symbol else symbol.split('/')[1]
            except:
                print(symbol)
        else:
            root = squality = bass = 'None'

        try:
            triad_type = quality_conversion_dict[squality]
        except:
            print('Error: invalid quality', squality)
            exit(1)

        return root, squality, bass, triad_type

    def structure_data(data):
        '''prepare for converting data to structured array'''
        new_data = []
        for row in data:
            onset = float(row[0]) if (isinstance(row[0], int) or isinstance(row[0], float)) else float(Fraction(row[0]))
            duration = float(row[1])
            symbol = row[2]
            root, squality, bass, triad_type = split_symbol(symbol) # split chord symbol into root, quality, bass, and triad type
            color = row[3]
            voicing = row[4]
            tonality = row[5]
            roman = row[6]
            degree1 = str(int(row[7])) if not isinstance(row[7], str) else row[7]
            degree2 = str(int(row[8])) if not isinstance(row[8], str) else row[8]
            rquality = row[9]
            inversion = str(int(row[10])) if not isinstance(row[10], str) else row[10]
            phrase = row[11]
            measure = row[12]
            rhythm = row[13]
            meter = row[14]

            new_data.append((onset, duration,
                             symbol, root, squality, bass, triad_type, color, voicing,
                             tonality, roman, degree1, degree2, rquality, inversion, phrase, measure, rhythm, meter))

        return(new_data)

    # read dataset
    corpus = {}
    dt = [('onset', float), ('duration', float),
          ('symbol', object), ('root', object), ('squality', object), ('bass', object), ('triad_type', object), ('color', object), ('voicing', object),
          ('tonality', object), ('roman', object), ('degree1', object), ('degree2', object), ('rquality', object), ('inversion', object),
          ('phrase', object), ('measure', int), ('rhythm', float), ('meter', object)] # pre-defined data type
    print('Reading dataset...')
    for i in range(1, 51): # 50 pieces
        print('piece', i)
        file_dir = data_dir + str(i) + '.xlsx'
        with xlrd.open_workbook(file_dir) as workbook:
            worksheet = workbook.sheets()[0]
            data = [[x.value for x in worksheet.row_slice(row)] for row in range(1, worksheet.nrows)]
        data = structure_data(data)
        corpus[str(i)] = np.array(data, dtype=dt)

    return corpus

def show_statistics(corpus, plot_figure=True):
    print("Statistics of Jazzification labels:")
    print('number of pieces =', len(corpus.keys()))
    print('number of phrases =', sum([len(set(itertools.chain.from_iterable([phrase_name.split(',') for phrase_name in chords['phrase']]))) for chords in corpus.values()]))
    number_of_labels = [(item[0], len(item[1])) for item in corpus.items()]
    print('number of chord in each piece =', number_of_labels)
    print('max =', max(number_of_labels, key=lambda x: x[1]), '  min =', min(number_of_labels, key=lambda x: x[1]))
    all_labels = np.concatenate(list(corpus.values()))
    all_notes = itertools.chain.from_iterable([voicing[1:-1].split(',') for voicing in all_labels['voicing'] if voicing != 'None'])
    all_note_numbers = [note_name_to_number(note_name) for note_name in all_notes]
    print('lowest pitch = ', min(all_note_numbers), '  highest pitch =', max(all_note_numbers))
    print('number of chords =', len(all_labels))
    print('number of inverted chords =', len([label for label in all_labels if label['inversion'] != '0' and label['inversion'] != 'None']))
    print('number of secondary chords =', len([label for label in all_labels if label['degree1'] != '1']))
    print('roots =', sorted(set(all_labels['root'])))
    print('basses =', sorted(set(all_labels['bass'])))
    print('symbol qualities =', rate_keys(all_labels['squality'], sorted_by_rate=True))
    print('roamn qualities =', rate_keys(all_labels['rquality'], sorted_by_rate=True))
    print('colors =', sorted(set(all_labels['color'])))
    print('tonality =', sorted(set(all_labels['tonality'])))
    print('1st degree =', sorted(set(all_labels['degree1'])))
    print('2nd degree =', sorted(set(all_labels['degree2'])))
    print('roman qualities =', sorted(set(all_labels['rquality'])))
    print('inversion =', sorted(set(all_labels['inversion'])))

    if plot_figure:
        # plot reduced chord symbol qualities
        squality_reduced = [re.sub("[\(\[].*?[\)\]]", "", x) for x in all_labels['squality']]
        squality_rate_list = rate_keys(squality_reduced, sorted_by_rate=True)
        ticks = [t[0] for t in squality_rate_list]
        values = [t[1] for t in squality_rate_list]
        fig, ax = plt.subplots()
        ax.set_title('Jazzification_Chord Symbol Qualities (reduced)')
        plt.bar(np.arange(len(ticks)), values, width=0.5)
        plt.xticks(np.arange(len(ticks)), ticks, rotation='vertical', fontsize=20)
        for i, v in enumerate(values):
            plt.text(x=i, y=v + 18, s=v, size=20, horizontalalignment='center')
        plt.yticks([])
        plt.xlabel('Chord Quality', fontsize=24)
        plt.ylabel('Frequency', fontsize=24)

        # plot colorings
        coloring = [replace_extended_colors(x) for x in all_labels['color']]
        coloring_rate_list = rate_keys(coloring, sorted_by_rate=True)
        ticks = [t[0] for t in coloring_rate_list if t[0] != 'None' and t[1] > 20]
        values = [t[1] for t in coloring_rate_list if t[0] != 'None' and t[1] > 20]
        fig, ax = plt.subplots()
        ax.set_title('Jazzification_Coloring')
        plt.bar(np.arange(len(ticks)), values, width=0.6)
        plt.xticks(np.arange(len(ticks)), ticks, rotation='vertical', fontsize=20)
        for i, v in enumerate(values):
            plt.text(x=i, y=v + 5, s=v, size=20, horizontalalignment='center')
        plt.yticks([])
        plt.xlabel('Coloring', fontsize=24)
        plt.ylabel('Frequency', fontsize=24)
        plt.show()

def phrasing(corpus):
    corpus_phrasing = dict.fromkeys(corpus)
    for op, labels in corpus.items():
        phrases = sorted(set(itertools.chain.from_iterable([x.split(',') for x in labels['phrase']])))
        phrases = {k:[] for k in phrases}
        for key in phrases.keys():
            for label in labels:
                if key in label['phrase']:
                    phrases[key].append(label)
        corpus_phrasing[op] = phrases

    num_phrases = [(k, len(v)) for k, v in corpus_phrasing.items()]
    print('number of phrases in each piece =', num_phrases)
    print('max =', max(num_phrases, key=lambda t: t[1]), ', min =', min(num_phrases, key=lambda t: t[1]))
    print('total number of phrases  =', sum([t[1] for t in num_phrases]))

    return corpus_phrasing # {'op': {'phrase':[label...]}, ... }

def extract_annotations(corpus_phrasing, transform=True):
    '''extract the annotations required for the chord jazzification task
        if transform==True, convert string labels to numerical numbers'''

    # note name categories
    note_name_dict = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    accidental_dict = {'': 0, '#': 1, 'b': -1, 'x': 2, 'd': -2}
    note_name_conversion_dict = {rk + ak: (rv + av) % 12 for rk, rv in note_name_dict.items() for ak, av in accidental_dict.items()}
    triad_type2int_dict = {'M': 0, 'm': 1, 'a': 2, 'd': 3, 'None': 4}

    def transform_label(label):
        ''' convert string to numerical number (except 'None')'''
        duration = label['duration']
        root = label['root']
        triad_type = label['triad_type']
        bass = label['bass']
        voicing = label['voicing']
        if root != 'None':
            new_root = note_name_conversion_dict[root]
            new_triad_type = triad_type2int_dict[triad_type]
            new_bass = note_name_conversion_dict[bass]
            voicing_split = voicing[1:-1].split(',')
            new_voicing = [note_name_to_number(note_name) for note_name in voicing_split] # convert note name to midi number
        else:
            new_root = new_triad_type = new_bass = new_voicing = 'None'
        return (new_root, new_triad_type, duration, new_bass, new_voicing)

    target_annotations = ['duration', 'root', 'triad_type', 'bass', 'voicing']
    corpus_phrasing_new = {}
    for op, phrase_dict in corpus_phrasing.items():
        corpus_phrasing_new[op] = {}
        for phrase_name, labels in phrase_dict.items():
            if transform:
                corpus_phrasing_new[op][phrase_name] = [transform_label(label[target_annotations]) for label in labels]
            else:
                corpus_phrasing_new[op][phrase_name] = [label[target_annotations] for label in labels]

    return corpus_phrasing_new


def augment_data(corpus_phrasing_reduced):
    def shift_label(label, shift):
        # (new_root, new_triad_type, duration, new_bass, new_voicing)
        if shift != 0:
            root = label[0]
            triad_type = label[1]
            duration = label[2]
            bass = label[3]
            voicing = label[4]
            if root != 'None':
                root_shift = (root + shift)%12
                bass_shift = (bass + shift)%12
                voicing_shift = [number + shift for number in voicing]
            else:
                root_shift = bass_shift = voicing_shift = 'None'
            return (root_shift, triad_type, duration, bass_shift, voicing_shift)

        else:
            return label

    corpus_phrasing_reduced_aug = {}
    # dt = [('triad_id', int), ('duration', float), ('bass_mod', int), ('treble_chroma', object), ('pianoroll_vector', object)]
    for shift in range(-4, 6):
        corpus_phrasing_reduced_aug['shift_' + str(shift)] = {}
        for op, phrase_dict in corpus_phrasing_reduced.items():
            corpus_phrasing_reduced_aug['shift_' + str(shift)][op] = {}
            for phrase_i, labels in phrase_dict.items():
                corpus_phrasing_reduced_aug['shift_' + str(shift)][op][phrase_i] = [shift_label(label, shift) for label in labels]


            # roots = [(chord['root'] + shift)%12 if chord['root'] != 12 else 12 for chord in chords]
            # durations = [chord['duration'] for chord in chords]
            # triad_types = [chord['triad_type'] for chord in chords]
            # triad_id = [r + 12*t if r != 12 else 48 for r,t in zip(roots, triad_types)]
            # # print(triad_id.index(48))
            # # print(roots[triad_id.index(48)])
            # # print(triad_types[triad_id.index(48)])
            # # quit()
            # voicings_number = [[number + shift for number in chord['voicing']] if chord['voicing'] != 'None' else 'None' for chord in chords]
            # voicings_expand = structure_voicing(voicings_number)
            # # print(chords[0])
            # # print(roots[0])
            # # print(triad_types[0])
            # # print(voicings_number[0])
            # # print(voicings_expand[0])
            # # quit()
            # # print(list(zip(roots, triad_types, voicings_expand))[0])
            #
            # data_dict_aug[piece][shift] = np.array([(t,) + (d,) + v for t, d, v in zip(triad_id, durations, voicings_expand)], dtype=dt)

    return corpus_phrasing_reduced_aug

def transform_data(Jazzification_corpus_aug):
    '''transform data into input/output format'''
    def transform (data):
        '''data = (root, triad_type, duration, bass, voicing)'''
        root = data[0]
        triad_type = data[1]
        duration = data[2]
        bass = data[3]
        voicing = data[4]
        if root != 'None':
            pianoroll_vector = np.array([0 if i not in [number - 21 for number in voicing] else 1 for i in range(88)], dtype=np.int32) # A0-C8 (= 21-108)
            treble_part = voicing[1:]
            treble_chroma = np.array([0 if i not in [number%12 for number in treble_part] else 1 for i in range(12)], dtype=np.int32)
        else:
            root = 12
            triad_type = 4
            bass = 12
            pianoroll_vector = np.zeros(88, dtype=np.int32)
            treble_chroma = np.zeros(12, dtype=np.int32)

        return (root, triad_type, duration, bass, treble_chroma, pianoroll_vector)

    Jazzification_corpus_aug_new = {}
    for shift, op_dict in Jazzification_corpus_aug.items():
        Jazzification_corpus_aug_new[shift] = {}
        for op, phrase_dict in op_dict.items():
            Jazzification_corpus_aug_new[shift][op] = {}
            for phrase_id, phrase in phrase_dict.items():
                Jazzification_corpus_aug_new[shift][op][phrase_id] = [transform(x) for x in phrase]

    return Jazzification_corpus_aug_new

def replace_extended_colors(color):
    rep = {"9": "2", "11": "4", "13": "6"} # define desired replacements here
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], color)

def reshape_data(Jazzification_corpus_aug_new, n_steps):

    def pad(phrase, n_steps):
        # padding: root = 12, triad_type = 4, duration = 0, bass = 12, trable_chorma = zero chroma, piano_vector = zero_vector
        pad = (12, 4, 0, 12, np.zeros(12, dtype=np.int32), np.zeros(88, dtype=np.int32))
        return phrase + [pad for _ in range(n_steps-len(phrase))]

    dt = [('root', np.int32), ('triad_type', np.int32), ('duration', np.float32), ('bass', np.int32), ('treble', object), ('piano_vector', object)]
    # for shift, op_dict in Jazzification_corpus_aug_new.items():
    #     for op, phrase_dict in op_dict.items():
    #         Jazzification_corpus_aug_new[shift][op] = np.array([pad(phrase, n_steps) for phrase_id, phrase in phrase_dict.items()], dtype=dt)
    for shift, op_dict in Jazzification_corpus_aug_new.items():
        # for phrase_dict in op_dict.values():
        Jazzification_corpus_aug_new[shift] = np.array([pad(phrase, n_steps) for phrase_dict in op_dict.values() for phrase in phrase_dict.values()], dtype=dt)

    return Jazzification_corpus_aug_new

def voicing2chorma(voicing):
    if voicing != 'None':
        voicing_number = [note_name_to_number(note_name) for note_name in voicing[1:-1].split(',')]
        chroma = np.array([0 if i not in [number % 12 for number in voicing_number] else 1 for i in range(12)], dtype=np.int32)
    else:
        chroma = np.zeros(12, dtype=np.int32)
    return chroma

def generate_midi_instance(piece, outputdir, qpm=120, play_midi=True, show_pianoroll=False):
    '''Generate midi with qpm (quarter note per minute) = 120
        wrote midi file using pretty_midi
        play midi using music21'''

    print('generate midi...')

    def plot_pianoroll(pianoroll):
        plt.imshow(pianoroll[21:109,:], aspect='auto', extent=[0, pianoroll.shape[1], 108, 21], cmap='gray_r')
        plt.gca().invert_yaxis()
        plt.xlabel('Time')
        plt.ylabel('Note Name')
        plt.yticks(range(21,109), [pm.note_number_to_name(x) for x in range(21,109)], fontsize=5)
        plt.show()

    # Create a PrettyMIDI object
    chord_progression = pm.PrettyMIDI()

    # Create an Instrument instance for a cello instrument
    program = pm.instrument_name_to_program('Bright Acoustic Piano')
    instrument = pm.Instrument(program=program)

    onset = 0.0
    for chord in piece:
        # Create a note instance
        duration = chord['duration'] * (60/qpm)
        voicing = chord['voicing'].replace("(", "").replace(")", "")
        note_names = voicing.split(',')
        if note_names[0] != 'None': # not 'None' chord
            for note_name in note_names:
                if ('x' not in note_name) and ('d' not in note_name): # without double accidentals
                    note_number = pm.note_name_to_number(note_name)
                else: # with double accidentals
                    note_number = pm.note_name_to_number(note_name[0] + note_name[-1])
                    note_number = note_number + 2 if 'x' in note_name else note_number - 2

                note = pm.Note(velocity=100, pitch=note_number, start=onset, end=onset+duration)
                # Add it to the instrument
                instrument.notes.append(note)
        onset += duration

    # Add the instrument to the PrettyMIDI object
    chord_progression.instruments.append(instrument)
    # Write out the MIDI data
    chord_progression.write(outputdir)
    print('output midi file as %s' % outputdir)

    if play_midi:
        # Play the midi file
        print('play midi...')
        stream = music21.converter.parse(outputdir)
        stream.show('midi')

    if show_pianoroll:
        # Represent the piece as a chromagram
        pianoroll = instrument.get_piano_roll(fs=qpm/60)
        plot_pianoroll(pianoroll)

def main():

    # Read the dataset
    data_dir = "Jazzification_Dataset\\"
    corpus = read_dataset(data_dir) # corpus = {'op': annotations, ...}

    # Show statistics of the dataset
    show_statistics(corpus, plot_figure=False)

    # # Generate midi
    # generate_midi_instance(corpus['1], 'example.mid', qpm=120, play_midi=True, show_pianoroll=False)

    # Group labels belonging to the same phrase
    corpus_phrasing = phrasing(corpus) # {'op': {'phrase':[label...]}, ... }
    chords_per_phrase = [len(phrase) for phrase_dict in corpus_phrasing.values() for phrase in phrase_dict.values()]
    max_number_of_steps = max(chords_per_phrase) # max number of steps in the chord progressions
    # print('number of chords per phrase', sorted(set(chords_per_phrase)))
    print('max steps =', max_number_of_steps)

    # Extract the annotations required for the chord jazzification task
    chord_jazzification_corpus = extract_annotations(corpus_phrasing, transform=True)

    # Dta augmentation (transpose the pieces from 4 semitones down to 5 semitones up)
    chord_jazzification_corpus_aug = augment_data(chord_jazzification_corpus) # {'shift_x': {'op': {'phrase': ...}}}

    # Transform data into input/output format
    chord_jazzification_corpus_aug_new = transform_data(chord_jazzification_corpus_aug)

    # Reshape data for training
    training_data = reshape_data(chord_jazzification_corpus_aug_new, n_steps=max_number_of_steps)
    training_data = np.concatenate([x for x in training_data.values()], axis=0) # shape = [n_sequences, n_steps]
    print('Training Data Info:')
    print('training_data.shape =', training_data.shape, '# [n_sequences, n_steps], with fields [\'root\', \'triad_type\', \'duration\', \'bass\', \'treble\', \'piano_vector\']')
    print('root: 0-11 for C-B; 12 for \'None\' and padding')
    print('triad_type: 0-3 for M, m, a, d; 4 for \'None\' and padding')
    print('duration: float; 0 for padding')
    print('bass: 0-11 for C-B; 12 for \'None\' and padding')
    print('treble: chroma vector; all zeros for \'None\' and padding')
    print('piano_vector: 88-d vector; all zeros for \'None\' and padding')

    # Save preprocessed data
    save_dir = 'chord_jazzification_training_data.pickle'
    with open(save_dir, 'wb') as save_file:
        pickle.dump(training_data, save_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('preprocessing completed.')


if __name__ == '__main__':

    main()