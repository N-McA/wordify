
from pathlib import Path
import pyFNR


def load_word_list(data_loc):
    data_loc = Path(data_loc)
    words = []
    with (data_loc / 'filtered.txt').open() as f:
        for line in f:
            words.append(line.strip())
    return words


def cum_prod(xs):
    r = [1]
    for x in xs:
        r.append(x*r[-1])
    return r


def cum_sum(xs):
    r = [0]
    for x in xs:
        r.append(x+r[-1])
    return r

def to_variable_base(n, cum_ranges):
    result = []
    if n >= cum_ranges[-1]:
        raise ValueError(
                'Cannot fit {} into this variable base'.format(
                    n))
    for q in reversed(cum_ranges):
        divider, remainder = divmod(n, q)
        result.append(divider)
        n = remainder
    return result[1:]


class Encoder:
    def __init__(self, *, word_list, capacity_in_bits, n_words):
        self.word_list = word_list
        self.word_to_index = {w: i for i, w in enumerate(word_list)}
        self.capacity_in_bits = capacity_in_bits
        self.n_words = n_words
        
        self.larger_n_bits = (len(self.word_list) ** self.n_words).bit_length() - 1
        if self.larger_n_bits <= self.capacity_in_bits:
            raise ValueError('Not enough capacity, more words needed.')
            
        self.n_pad_bits = self.larger_n_bits - self.capacity_in_bits
            
        # Determines the bits that we can pad out
        self.pad_mask = (int(
            '0b' 
            + ('0' * self.n_pad_bits) 
            + ('1' * self.capacity_in_bits), base=2))
        
        self.fnr = fnr = pyFNR.FNR2(
            domain=(len(self.word_list) ** self.n_words),
            key='0000000000000000',
        )
        
        self._cum_ranges = cum_prod([len(self.word_list)] * self.n_words)
        
    def _to_words(self, n: int):
        indices = to_variable_base(n, self._cum_ranges)
        return [self.word_list[idx] for idx in indices]
    
    def _from_words(self, words):
        idxs = reversed([self.word_to_index[w] for w in words])
        return sum([idx*b for idx, b in zip(idxs, self._cum_ranges)])

    def encodings(self, n: int):
        assert n.bit_length() <= self.capacity_in_bits
        for v in range(0, 2**self.n_pad_bits - 1):
            padded = n ^ (v << self.capacity_in_bits)
            bijected = self.fnr.encrypt(padded)
            yield self._to_words(bijected)

    def decode(self, words: list):
        assert len(words) == self.n_words
        n = self._from_words(words)
        bijected = self.fnr.decrypt(n)
        return bijected & self.pad_mask
