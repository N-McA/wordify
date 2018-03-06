
from flask import Flask
from flask import jsonify
app = Flask(__name__)

import hashlib
import pickle
from functools import lru_cache

from wordify import encoder
from wordify.constants import config

import keras
import numpy as np

words = encoder.load_long_word_list(config.data_loc)

enc = encoder.Encoder(
    word_list=words,
    capacity_in_bits=config.capacity_in_bits,
    n_words=config.n_words,
)

def sha(s):
    m = hashlib.sha256()
    m.update(s.encode('utf8'))
    return m.digest()[:5]

def ten_digit_hash(s):
    return int(str(int.from_bytes(sha(s), 'little'))[:10])

def dash_split(n):
    s = str(n).zfill(10)
    return '-'.join([s[i:i+2] for i in range(0, len(s)-2, 2)])

def id_from_email(s):
    s = s.split('@')[0]
    return ten_digit_hash(s)


START = '<S>'
END = '<E>'
model = keras.models.load_model(config.data_loc / 'language_model_weights/weights.hdf5')
with open('/home/nm583/sfm_data/definition_char_to_int.pickle', 'rb') as f:
    char_to_int = pickle.load(f)
int_to_char = {i:c for c, i in char_to_int.items()}

def log_prob(seq):
    x = [char_to_int[START]] + [char_to_int[c] for c in seq] + [char_to_int[END]]
    probs = model.predict(np.array([x]))[0]
    log_probs = np.log(probs)
    t = 0
    for i, idx in enumerate(x[1:]):
        t += log_probs[i, idx]
    return t

def log_prob_encoding(e):
    return log_prob(' '.join(e))


@lru_cache(2**13)
def get_codes_worker(email):
    uid = id_from_email(email)
    encodings = sorted(enc.encodings(uid), key=log_prob_encoding)
    return {
        'numeric': dash_split(uid),
        'worst_model': ' '.join(encodings[0]),
        'best_model': ' '.join(encodings[-1]),
    }


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/codes/<string:email>')
def get_codes(email):
    return jsonify(get_codes_worker(email))