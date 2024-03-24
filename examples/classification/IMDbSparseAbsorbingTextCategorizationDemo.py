import argparse
import logging
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.classification.vanilla_classifier import TMClassifier

from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[],
        absorbed=[],
        unallocated=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )


def main(args):
    experiment_results = metrics(args)

    _LOGGER.info("Preparing dataset")
    train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
    train_x, train_y = train
    test_x, test_y = test

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    _LOGGER.info("Preparing dataset.... Done!")

    _LOGGER.info("Producing bit representation...")

    id_to_word = {value: key for key, value in word_to_id.items()}

    training_documents = []
    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id].lower())

        training_documents.append(terms)

    testing_documents = []
    for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
            terms.append(id_to_word[word_id].lower())

        testing_documents.append(terms)

    vectorizer_X = CountVectorizer(
        tokenizer=lambda s: s,
        token_pattern=None,
        ngram_range=(1, args.max_ngram),
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents)
    Y_train = train_y.astype(np.uint32)

    X_test = vectorizer_X.transform(testing_documents)
    Y_test = test_y.astype(np.uint32)
    _LOGGER.info("Producing bit representation... Done!")

    _LOGGER.info("Selecting Features....")

    SKB = SelectKBest(chi2, k=args.features)
    SKB.fit(X_train, Y_train)

    selected_features = SKB.get_support(indices=True)
    X_train = SKB.transform(X_train).astype(np.uint32)
    X_test = SKB.transform(X_test).astype(np.uint32)

    _LOGGER.info("Selecting Features.... Done!")

    tm = TMClassifier(
        args.num_clauses,
        args.T,
        args.s,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        clause_drop_p=args.clause_drop_p,
        literal_insertion_state=args.literal_insertion_state,
        literal_sampling=args.literal_sampling,
        absorbing=args.absorbing
    )

    for e in range(args.epochs):

        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, Y_train)
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(X_test) == Y_test).mean()
        experiment_results["test_time"].append(benchmark2.elapsed())
        experiment_results["accuracy"].append(int(result))

        absorbed = 0.0
        unallocated = 0
        for i in range(2):
            for j in range(args.num_clauses):
                absorbed += 1.0 - (tm.number_of_include_actions(i, j) + tm.number_of_exclude_actions(i, j)) / (
                            X_train.shape[1] * 2)
                unallocated += tm.number_of_unallocated_literals(i, j)
        absorbed = 100 * absorbed / (2 * args.num_clauses)

        experiment_results["absorbed"].append(int(absorbed))
        experiment_results["unallocated"].append(int(unallocated))

        _LOGGER.info("#%d Accuracy: %.2f%% Absorbed: %.2f%% Unallocated: %d Training: %.2fs Testing: %.2fs" % (
            e + 1,
            result,
            absorbed,
            unallocated,
            benchmark1.elapsed(),
            benchmark2.elapsed()
        ))

    return experiment_results

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=10000, type=int)
    parser.add_argument("--T", default=8000, type=int)
    parser.add_argument("--s", default=2.0, type=float)
    parser.add_argument("--platform", default="CPU_sparse", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--clause_drop_p", default=0.75, type=float)
    parser.add_argument("--max-ngram", default=2, type=int)
    parser.add_argument("--features", default=5000, type=int)
    parser.add_argument("--imdb-num-words", default=5000, type=int)
    parser.add_argument("--imdb-index-from", default=2, type=int)
    parser.add_argument("--literal-insertion-state", default=100, type=int)
    parser.add_argument("--literal-sampling", default=0.1, type=float)
    parser.add_argument("--absorbing", default=90, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":

    results = main(default_args())
    _LOGGER.info(results)
