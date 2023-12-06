from typing import Dict
import numpy as np
from tmu.data import TMUDataset
from tmu.data.utils.downloader import get_file
import json
import pathlib


class IMDB(TMUDataset):

    def __init__(
            self,
            path="imdb.npz",
            num_words=None,
            skip_top=0,
            maxlen=None,
            seed=113,
            start_char=1,
            oov_char=2,
            index_from=3,
            **kwargs,
    ):
        super().__init__()
        self.path = path
        self.num_words = num_words
        self.skip_top = skip_top
        self.maxlen = maxlen
        self.seed = seed
        self.start_char = start_char
        self.oov_char = oov_char
        self.index_from = index_from
        self.kwargs = kwargs

    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        origin_folder = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        )

        path = get_file(
            "imdb.npz",
            origin=origin_folder + "imdb.npz",
            file_hash=(  # noqa: E501
                "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f"
            ),
        )

        with np.load(path, allow_pickle=True) as f:
            x_train, labels_train = f["x_train"], f["y_train"]
            x_test, labels_test = f["x_test"], f["y_test"]

        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        rng.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        if self.start_char is not None:
            x_train = [[self.start_char] + [w + self.index_from for w in x] for x in x_train]
            x_test = [[self.start_char] + [w + self.index_from for w in x] for x in x_test]
        elif self.index_from:
            x_train = [[w + self.index_from for w in x] for x in x_train]
            x_test = [[w + self.index_from for w in x] for x in x_test]

        if self.maxlen:
            x_train, labels_train = _remove_long_seq(self.maxlen, x_train, labels_train)
            x_test, labels_test = _remove_long_seq(self.maxlen, x_test, labels_test)
            if not x_train or not x_test:
                raise ValueError(
                    "After filtering for sequences shorter than maxlen="
                    f"{str(self.maxlen)}, no sequence was kept. Increase maxlen."
                )

        xs = x_train + x_test
        labels = np.concatenate([labels_train, labels_test])

        if not self.num_words:
            self.num_words = max(max(x) for x in xs)

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if self.oov_char is not None:
            xs = [
                [w if (self.skip_top <= w < self.num_words) else self.oov_char for w in x]
                for x in xs
            ]
        else:
            xs = [[w for w in x if self.skip_top <= w < self.num_words] for x in xs]

        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx], dtype="object"), labels[:idx]
        x_test, y_test = np.array(xs[idx:], dtype="object"), labels[idx:]

        return dict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    def get_word_index(self, path="imdb_word_index.json"):
        """Retrieves a dict mapping words to their index in the IMDB dataset.

        Args:
            path: where to cache the data (relative to `~/.keras/dataset`).

        Returns:
            The word index dictionary. Keys are word strings, values are their
            index.

        Example:

        ```python
        # Use the default parameters to keras.datasets.imdb.load_data
        start_char = 1
        oov_char = 2
        index_from = 3
        # Retrieve the training sequences.
        (x_train, _), _ = keras.datasets.imdb.load_data(
            start_char=start_char, oov_char=oov_char, index_from=index_from
        )
        # Retrieve the word index file mapping words to indices
        word_index = keras.datasets.imdb.get_word_index()
        # Reverse the word index to obtain a dict mapping indices to words
        # And add `index_from` to indices to sync with `x_train`
        inverted_word_index = dict(
            (i + index_from, word) for (word, i) in word_index.items()
        )
        # Update `inverted_word_index` to include `start_char` and `oov_char`
        inverted_word_index[start_char] = "[START]"
        inverted_word_index[oov_char] = "[OOV]"
        # Decode the first sequence in the dataset
        decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
        ```
        """
        origin_folder = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        )
        path = get_file(
            pathlib.Path(path),
            origin=origin_folder + "imdb_word_index.json",
            file_hash="bfafd718b763782e994055a2d397834f",
        )
        with open(path + ".json") as f:
            return json.load(f)

    def _transform(self, name, dataset):
        return dataset
