
from wordify import encoder


def test_encoder():
    words = encoder.load_word_list('./data')
    enc = encoder.Encoder(
        word_list=words,
        capacity_in_bits=(10**10 - 1).bit_length(),
        n_words=3,
    )

    ssn = 9999999999
    for e in enc.encodings(ssn):
        assert enc.decode(e) == ssn
        
    ssn = 0
    for e in enc.encodings(ssn):
        assert enc.decode(e) == ssn
        
    ssn = int('9'*10, base=10)
    for e in enc.encodings(ssn):
        assert enc.decode(e) == ssn
    