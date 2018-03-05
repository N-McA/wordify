
import pickle
from pathlib import Path
import re

import numpy as np

import keras
from keras.models import Model
from keras.layers import GRU, Dense, Input, Embedding
from keras_pickle_wrapper import KerasPickleWrapper


class CharNNWordEmbedder:
    def __init__(self, charnn_model, char_to_int):
        self._charnn_model = KerasPickleWrapper(charnn_model)
        self.char_to_int = char_to_int
        
    @property
    def charnn_model(self):
        return self._charnn_model()
    
    def embed_string(self, s):
        input_ints = [char_to_int[c] for c in s]
        input_seq = np.array([input_ints])
        decoder_state = self.charnn_model.predict(input_seq)
        return decoder_state
    
    def embed_strings(self, strings):
        input_intss = [[char_to_int[c] for c in s] for s in strings]
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

    from keras.preprocessing.sequence import pad_sequences

    xs = np.array(xs)
    ys = np.array(ys)

    def train_gen():
        indices = np.arange(len(xs))
        while True:
            selected = np.random.choice(indices, size=64, replace=False)
            encoder_inputs = pad_sequences(xs[selected], padding='post')
            targets = ys[selected]
            target_seqs = keras.utils.to_categorical(
                pad_sequences(targets, padding='post'), num_classes=len(phoneme_to_int)
            )
            decoder_target_next_outputs = target_seqs[:, 1:]  
            decoder_prev_ideal_outputs = target_seqs[:, :-1]
            # [a, b, c, <eos>]
            # [<start>, a, b, c]
            yield [encoder_inputs, decoder_prev_ideal_outputs], decoder_target_next_outputs


    opt = keras.optimizers.Adam(amsgrad=True)
    model.compile(opt, loss='categorical_crossentropy')

    model.fit_generator(
        train_gen(),
    #     steps_per_epoch=len(xs)//64,
        steps_per_epoch=50,
    )

    return CharNNWordEmbedder(
        char_to_int=char_to_int,
        charnn_model=encoder_model,
    )


def main():
    charnn_word_embedder = train_model()
    with open('/home/nm583/phonetic_charnn.pickle', 'wb') as f:
        picle.dump(charnn_word_embedder, f)
        

if __name__ == '__main__':
    main()
    