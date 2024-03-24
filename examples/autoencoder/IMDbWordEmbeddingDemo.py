import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from tmu.data import IMDB
from tmu.tools import BenchmarkTimer
import logging
import os
import argparse

os.environ["CUDA_PROFILE"] = "1"

_LOGGER = logging.getLogger(__name__)

target_words = [
    'awful',
    'terrible',
    'lousy',
    'abysmal',
    'crap',
    'outstanding',
    'brilliant',
    'excellent',
    'superb',
    'magnificent',
    'marvellous',
    'truck',
    'plane',
    'car',
    'cars',
    'motorcycle',
    'scary',
    'frightening',
    'terrifying',
    'horrifying',
    'funny',
    'comic',
    'hilarious',
    'witty'
]

def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        precision=[],
        recall=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    # load IMDB dataset
    dataloader = IMDB(num_words=args.NUM_WORDS, index_from=args.INDEX_FROM)
    dataset = dataloader.get()
    train_x = dataset["x_train"]
    train_y = dataset["y_train"]
    test_x = dataset["x_test"]
    test_y = dataset["y_test"]

    # get word to indice and id_to_word mappings
    word_to_id = dataloader.get_word_index()
    word_to_id = {k: (v + args.INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}

    _LOGGER.info("Producing bit representation...")

    def produce_documents(dsx, dsy):
        docs = []
        for i in range(dsy.shape[0]):
            terms = [id_to_word[word_id].lower() for word_id in dsx[i]]
            docs.append(terms)
        return docs

    training_documents = produce_documents(train_x, train_y)
    testing_documents = produce_documents(test_x, test_y)

    # Create vectorizer
    def tokenizer(s):
        return s

    vectorizer_X = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents)
    feature_names = vectorizer_X.get_feature_names_out()

    output_active = np.empty(len(target_words), dtype=np.uint32)
    for i in range(len(target_words)):
        target_word = target_words[i]
        target_id = vectorizer_X.vocabulary_[target_word]
        output_active[i] = target_id

    tm = TMAutoEncoder(
        number_of_clauses=args.clauses,
        T=args.T,
        s=args.s,
        output_active=output_active,
        max_included_literals=args.max_included_literals,
        accumulation=args.accumulation,
        feature_negation=args.feature_negation,
        platform=args.device,
        output_balancing=args.output_balancing
    )

    benchmark_train = BenchmarkTimer()
    benchmark_test = BenchmarkTimer()
    benchmark_total = BenchmarkTimer()

    _LOGGER.info(f"Accuracy over {args.epochs} epochs:")
    for e in range(args.epochs):
        with benchmark_total:
            _LOGGER.info(f"Epoch {e + 1}")

            with benchmark_train:
                tm.fit(X_train, number_of_examples=args.number_of_examples)
            experiment_results["train_time"].append(benchmark_train.elapsed())

            with benchmark_test:
                _LOGGER.info("Calculating precision")
                precision = [tm.clause_precision(i, True, X_train, number_of_examples=500) for i in
                             tqdm(range(len(target_words)), desc="Precision")]

                _LOGGER.info("Calculating recall")
                recall = [tm.clause_recall(i, True, X_train, number_of_examples=500) for i in
                          tqdm(range(len(target_words)), desc="Recall")]

                experiment_results["precision"].append(precision)
                experiment_results["recall"].append(precision)

                _LOGGER.info("Clauses")
                for j in range(args.clauses):
                    clause_info = " ".join(
                        [f"{target_words[i]}:W{tm.get_weight(i, j)}:P{precision[i][j]:.2f}:R{recall[i][j]:.2f}" for i in
                         range(len(target_words))])
                    literals = ["{}{}({})".format("¬" if k >= tm.clause_bank.number_of_features else "",
                                                  feature_names[k % tm.clause_bank.number_of_features],
                                                  tm.clause_bank.get_ta_state(j, k)) for k in
                                range(tm.clause_bank.number_of_literals) if tm.get_ta_action(j, k) == 1]
                    _LOGGER.info(f"Clause #{j} {clause_info} {' ∧ '.join(literals)}")

                profile = np.array(
                    [np.where(tm.get_weights(i) >= args.clause_weight_threshold, tm.get_weights(i), 0) for i in
                     range(len(target_words))])
                similarity = cosine_similarity(profile)

                _LOGGER.info("Word Similarity:")
                for i in range(len(target_words)):
                    sorted_index = np.argsort(-similarity[i, :])
                    similarity_info = " ".join(
                        [f"{target_words[sorted_index[j]]}({similarity[i, sorted_index[j]]:.2f})" for j in
                         range(1, len(target_words))])
                    _LOGGER.info(f"{target_words[i]}: {similarity_info}")

            experiment_results["test_time"].append(benchmark_train.elapsed())

        _LOGGER.info(f"Total time: {benchmark_total.elapsed():.2f}s")
        _LOGGER.info(f"Training time: {benchmark_train.elapsed():.2f}s")
        _LOGGER.info(f"Testing time: {benchmark_test.elapsed():.2f}s")
    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clause_weight_threshold', type=int, default=0, help='Clause weight threshold')
    parser.add_argument('--number_of_examples', type=int, default=2500, help='Number of examples')
    parser.add_argument('--accumulation', type=int, default=25, help='Accumulation')
    parser.add_argument('--factor', type=int, default=4, help='Factor')
    parser.add_argument('--clauses', type=int, default=80, help='Clauses')  # factor*20
    parser.add_argument('--T', type=int, default=160, help='T')  # factor*40
    parser.add_argument('--s', type=float, default=5.0, help='S')
    parser.add_argument('--NUM_WORDS', type=int, default=10000, help='Number of words')
    parser.add_argument('--INDEX_FROM', type=int, default=2, help='Index from')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--device', type=str, default="CUDA", help='which device to use')
    parser.add_argument("--output_balancing", type=float, default=0.5, help="output balancing")
    parser.add_argument("--number_of_output_words", type=int, default=20, help="how many words")
    parser.add_argument("--max_included_literals", type=int, default=3, help="max included literals")
    parser.add_argument("--feature_negation", type=bool, default=True, help="feature negation")
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)
