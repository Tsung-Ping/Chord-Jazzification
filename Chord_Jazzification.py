from chord_jazzification_models import *
from collections import namedtuple


# Disables AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # cross_validate_coloring_model(data_dir, hp) # train the coloring model with cross-validation sets
    # cross_validate_voicing_model(data_dir, hp) # train the voicing model with cross-validation sets
    # cross_validate_end2end_chord_jazzification(data_dir, hp) # train the end-to-end chord jazzification model with cross-validation sets
    # train_chord_jazzification(data_dir, hp) # train the chord jazzification model with the whole dataset

    chord_jazzification_inference(hp, threshold=0.6, user_input=False) # generate jazzified chord sequences from the JAAH dataset using the chord jazzification model trained on the chord jazzificaion dataset

if __name__ == '__main__':

    # Hyperparameters
    inner_b_classes = 13 # 12 pitch classes + 1 'None'
    inner_p_classes = 12 # 12 pitch classes
    output_v_classes = 88
    n_steps = 67 # max number of steps in sequences
    n_units = 512 # number of hidden units
    drop = 0.5 # dropout rate
    batch_size = 30
    beta_ce_p = 4
    # beta_ce_v = 4
    end2end_beta_L2 = 1e-4
    coloring_beta_L2 = 5e-4
    voicing_beta_L2 = 1e-4
    beta_L2 = 1e-3
    initial_learning_rate = 1e-3
    training_steps = 15000
    n_in_succession = 10 # early stop if the performance has not improved for n_in_succession epoches

    data_dir = 'chord_jazzification_training_data.pickle'
    validation_set_id = 0 # cross validation set = {0, 1, 2, 3}
    with_em = True # if true, convert one-hot vector to dense vector
    with_mask = True # if true, using voicing mask
    sequential_model = 'mhsa' # 'blstm' #

    hyperparameters = namedtuple("Hyperparameters",
                                 ['inner_b_classes',
                                  'inner_p_classes',
                                  'output_v_classes',
                                  'n_steps',
                                  'n_units',
                                  'drop',
                                  'batch_size',
                                  'beta_ce_p',
                                  'end2end_beta_L2',
                                  'coloring_beta_L2',
                                  'voicing_beta_L2',
                                  'initial_learning_rate',
                                  'training_steps',
                                  'n_in_succession',
                                  'with_em',
                                  'with_mask',
                                  'sequential_model',
                                  'validation_set_id'])

    hp = hyperparameters(inner_b_classes=inner_b_classes,
                         inner_p_classes=inner_p_classes,
                         output_v_classes=output_v_classes,
                         n_steps=n_steps,
                         n_units=n_units,
                         drop=drop,
                         batch_size=batch_size,
                         beta_ce_p=beta_ce_p,
                         end2end_beta_L2=end2end_beta_L2,
                         coloring_beta_L2=coloring_beta_L2,
                         voicing_beta_L2=voicing_beta_L2,
                         initial_learning_rate=initial_learning_rate,
                         training_steps=training_steps,
                         n_in_succession=n_in_succession,
                         with_em=with_em,
                         with_mask=with_mask,
                         sequential_model=sequential_model,
                         validation_set_id=validation_set_id)


    main()




