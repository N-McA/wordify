
from pathlib import Path

data_loc = Path('data/')
with (data_loc / 'google-10000-english-usa-no-swears.txt').open() as f:
    for line_n, line in enumerate(f):
        if line_n < 30:
            continue
        word = line.strip()
        if len(word) < 3:
            continue
        if len(word) < 4 and line_n > 1000:
            continue
        print(word, flush=True)

