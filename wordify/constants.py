
from pathlib import Path
from types import SimpleNamespace
import os

script_loc = Path(os.path.dirname(os.path.realpath(__file__)))
root_loc = script_loc / '..'

# This is not ideal, but simplest way for now.

config = SimpleNamespace(**{
    'root_loc': root_loc,
    'data_loc': root_loc / 'data',
    'n_words': 3,
    'capacity_in_bits': (10**10 - 1).bit_length(),
})