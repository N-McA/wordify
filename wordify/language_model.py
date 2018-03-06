from pathlib import Path
import string
from unidecode import unidecode
import pickle
import random
import numpy as np

from wordify.constants import config
import wordify.callbacks as callbacks

import keras
from keras.models import Model
from keras.layers import GRU, Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))



def train_model():
    '''
    data_loc = Path('/local/scratch/nm583/urbandict-word-def.csv')
    punctuation_to_none = ''.maketrans({
        **{p: None for p in string.punctuation},
        **{d: None for d in string.digits},
        '\x03': None,
        '\x19': None
    })
    char_vocab = set()
    definitions = []
    with data_loc.open() as f:
        # Skip header
        next(f)
        for line_n, line in enumerate(f):
            definition = line.split(',', maxsplit=5)[-1][1:-1]
            definition = unidecode(definition)
            definition = definition.lower()
            definition = definition.translate(punctuation_to_none)
            # Split with no args does whitespace stripping + split.
            definition = ' '.join(definition.split())
            if len(definition) == 0:
                continue
            definitions.append(definition)
            char_vocab |= set(definition)

    np.random.shuffle(definitions)
    with open('/home/nm583/sfm_data/definitions.pickle', 'wb') as f:
        pickle.dump(definitions, f)
    '''
    
    with open('/home/nm583/sfm_data/definitions.pickle', 'rb') as f:
        definitions = pickle.load(f)

    TRAIN_PROPORTION = 0.85
    n_train = int(TRAIN_PROPORTION * len(definitions))
    train_defs = definitions[:n_train]
    val_defs = definitions[n_train:]

    START = '<S>'
    END = '<E>'
    '''
    char_vocab.add(START)
    char_vocab.add(END)
    char_to_int = {c: i for i, c in enumerate(sorted(list(char_vocab)))}
    int_to_char = {i: c for c, i in char_to_int.items()}
    
    with open('/home/nm583/sfm_data/definition_char_to_int.pickle', 'wb') as f:
        pickle.dump(char_to_int, f)
    '''
        
    with open('/home/nm583/sfm_data/definition_char_to_int.pickle', 'rb') as f:
        char_to_int = pickle.load(f)
    int_to_char = {i:c for c, i in char_to_int.items()}

    CHAR_EMBEDDING_SIZE = 128
    HIDDEN_REP_SIZE = 256

    encoder_inputs = Input(shape=[None])
    x = encoder_inputs
    x = Embedding(len(int_to_char), CHAR_EMBEDDING_SIZE)(x)
    output_states = GRU(HIDDEN_REP_SIZE, return_sequences=True)(x)
    output_probs = Dense(len(char_to_int), activation='softmax')(output_states)
    model = Model(encoder_inputs, output_probs)
    opt = keras.optimizers.Adam(amsgrad=True)
    model.compile(opt, loss='categorical_crossentropy')
    
    ngram_n = 3
    def ngramify(ls, ngram_n):
        return [ls[i:i+ngram_n] for i in range(len(ls) - ngram_n + 1)]

    def ngram_samples(defs, batch_size):
        i = 0
        while True:
            batch_strings = []
            while len(batch_strings) < batch_size:
                d = defs[i]
                try:
                    batch_strings.append(' '.join(random.choice(ngramify(d.split(), 3))))
                except IndexError as e:
                    pass
                i = (i + 1) % len(defs)
            yield batch_strings
        
        
    def padded_samples(defs, batch_size):
        for batch_strings in ngram_samples(defs, batch_size):
            batch_ints = [[char_to_int[c] for c in s] for s in batch_strings]
            s, e = char_to_int[START], char_to_int[END]
            for b in batch_ints:
                b.insert(0, s)
                b.append(e)
            seq_inputs = pad_sequences(batch_ints, padding='post', value=char_to_int[END])
            target_seqs = keras.utils.to_categorical(
                seq_inputs, num_classes=len(char_to_int)
            )
            yield seq_inputs[:, :-1], target_seqs[:, 1:]

    batch_size = 512
    model.fit_generator(
        padded_samples(train_defs, batch_size=batch_size),
        #steps_per_epoch=5000,
        steps_per_epoch=len(train_defs)//batch_size,
        epochs=100,
        validation_data=padded_samples(val_defs, batch_size),
        validation_steps=len(val_defs)//batch_size,
        #validation_steps=750,
        callbacks=[
                callbacks.NDJSONLoggingCallback(config.data_loc / 'language_model.log'),
                callbacks.ModelCheckpoint(
                    model=model, 
                    path=config.data_loc / 'language_model_weights', 
                    save_best_only=True,
                    verbose=True,
                ),
            ],
    )
    
if __name__ == '__main__':
    train_model()
