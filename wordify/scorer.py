
import pickle
from pathlib import Path
import re

import numpy as np

import keras
from keras.models import Model
from keras.layers import GRU, Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras_pickle_wrapper import KerasPickleWrapper

from wordify.constants import config
import wordify.callbacks as callbacks


class CharNNWordEmbedder:
    def __init__(self, charnn_model, char_to_int):
        self._charnn_model = KerasPickleWrapper(charnn_model)
        self.char_to_int = char_to_int
        
    @property
    def charnn_model(self):
        return self._charnn_model()
    
    def embed_string(self, s):
        input_ints = [self.char_to_int[c] for c in s]
        input_seq = np.array([input_ints])
        decoder_state = self.charnn_model.predict(input_seq)
        return decoder_state
    
    def embed_strings(self, strings):
        input_intss = [[self.char_to_int[c] for c in s] for s in strings]
        input_seqs = np.array(input_intss)
        input_seqs = pad_sequences(input_seqs, padding='post')
        decoder_states = self.charnn_model.predict(input_seqs)
        return decoder_states
    

def train_model():
    START = '<START>'
    EOS = '<EOS>'
    data_loc = Path('data/cmudict/')

    not_braces = re.compile('[^()]+')
    number_split = re.compile('([A-Z]+)([012])')

    phoneme_vocab = set()
    char_vocab = set()

    xs = []
    ys = []
    with (data_loc / 'cmudict.dict').open() as f:
        for line_n, line in enumerate(f):
            if '#' in line:
                # Skip anything with a comment associated.
                continue
            word_num, phonetics = line.split(' ', 1)
            word = not_braces.findall(word_num)[0]
            r = []
            for phoneme in phonetics.strip().split(' '):
                m = number_split.match(phoneme)
                if m:
                    r += list(m.groups())
                else:
                    r.append(phoneme)
            char_vocab.update(word)
            phoneme_vocab.update(r)

            xs.append(list(word))
            ys.append(r)
        

    char_vocab.add(START)
    char_vocab.add(EOS)
    phoneme_vocab.add(START)
    phoneme_vocab.add(EOS)

    phoneme_to_int = {p: i for i, p in enumerate(sorted(list(phoneme_vocab)))}
    char_to_int = {c: i for i, c in enumerate(sorted(list(char_vocab)))}

    int_to_char = {i: c for c, i in char_to_int.items()}
    int_to_phoneme = {i: p for p, i in phoneme_to_int.items()}

    for y in ys:
        y.insert(0, START)
        y.append(EOS)

    xs = [[char_to_int[c] for c in x] for x in xs]
    ys = [[phoneme_to_int[p] for p in y] for y in ys]

    CHAR_EMBEDDING_SIZE = 128
    HIDDEN_REP_SIZE = 512

    encoder_inputs = Input(shape=[None])
    x = encoder_inputs
    x = Embedding(len(int_to_char), CHAR_EMBEDDING_SIZE)(x)
    _, encoded_state = GRU(HIDDEN_REP_SIZE, return_state=True)(x)
    encoder_model = Model(encoder_inputs, encoded_state)

    
    decoder_inputs = Input(shape=(None, len(phoneme_vocab)))
    decoder_state_input = Input(shape=[HIDDEN_REP_SIZE])
    decoder_gru = GRU(HIDDEN_REP_SIZE, return_sequences=True, return_state=True)
    decoder_dense = Dense(len(phoneme_vocab), activation='softmax')


    decoder_train_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoded_state)
    decoder_train_outputs = decoder_dense(decoder_train_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_train_outputs)

    decoder_outputs, decoder_state = \
        decoder_gru(decoder_inputs, initial_state=decoder_state_input)

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input], [decoder_outputs, decoder_state])

    xs = np.array(xs)
    ys = np.array(ys)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    xs = xs[indices]
    ys = ys[indices]
    
    
    TRAIN_PROPORTION = 0.85
    n_train = int(len(xs) * TRAIN_PROPORTION)
    train_xs, train_ys = xs[:n_train], ys[:n_train]
    val_xs, val_ys = xs[n_train:], ys[n_train:]

    def padded_samples(xs, ys, batch_size):
        indices = np.arange(len(xs))
        while True:
            selected = np.random.choice(indices, size=batch_size, replace=False)
            encoder_inputs = pad_sequences(xs[selected], padding='post', value=char_to_int[EOS])
            targets = ys[selected]
            target_seqs = keras.utils.to_categorical(
                pad_sequences(targets, padding='post', value=char_to_int[EOS]), num_classes=len(phoneme_to_int)
            )
            decoder_target_next_outputs = target_seqs[:, 1:]  
            decoder_prev_ideal_outputs = target_seqs[:, :-1]
            # [a, b, c, <eos>]
            # [<start>, a, b, c]
            yield [encoder_inputs, decoder_prev_ideal_outputs], decoder_target_next_outputs


    opt = keras.optimizers.Adam(amsgrad=True)
    model.compile(opt, loss='categorical_crossentropy')
    
    
    with (config.data_loc / 'phonetic_char_to_int.pickle').open('wb') as f:
        pickle.dump(char_to_int, f)
    
    batch_size = 64
    model.fit_generator(
        padded_samples(train_xs, train_ys, batch_size),
        steps_per_epoch=len(xs)//batch_size,
        epochs=50,
        validation_data=padded_samples(val_xs, val_ys, batch_size),
        validation_steps=len(val_xs)//batch_size,
        callbacks=[
            callbacks.NDJSONLoggingCallback(config.data_loc / 'phonetic.log'),
            callbacks.ModelCheckpoint(
                model=encoder_model, 
                path=config.data_loc / 'phonetic_weights', 
                save_best_only=True,
                verbose=True,
            ),
        ],
    )
    
    return CharNNWordEmbedder(
        char_to_int=char_to_int,
        charnn_model=encoder_model,
    )


def get_phonetic_charnn():
    with (config.data_loc / 'phonetic_charnn.pickle').open('rb') as f:
        m = pickle.load(f)
    return m


def main():
    charnn_word_embedder = train_model()
    with (config.data_loc / 'phonetic_charnn.pickle').open('wb') as f:
        pickle.dump(charnn_word_embedder, f)
        

if __name__ == '__main__':
    main()
    
