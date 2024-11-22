import argparse
from math import sqrt
from pprint import pprint

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import  classification_report
from tmu.data import MNIST
from tmu.experimental.models.multioutput_classifier import TMCoalesceMultiOuputClassifier
from tqdm import tqdm

colors = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "yellow": [1, 1, 0],
    "cyan": [0, 1, 1],
    "magenta": [1, 0, 1],
    "white": [1, 1, 1],
}
num = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    0: "zero",
}


def check_prime(n):
    if n > 1:
        is_prime = True
        for i in range(2, int(sqrt(n)) + 1):
            if n % i == 0:
                is_prime = False
                break
        return is_prime
    else:
        return False


def add_color_mnist(x, y):
    n = x.shape[0]
    x = x.reshape(-1, 28, 28)
    nt = np.concatenate([np.ones(n // 2), np.zeros(n - (n // 2))])
    np.random.shuffle(nt)

    x_color = np.stack([x] * 3, axis=-1)

    tx = []
    for i in tqdm(range(n)):
        color_name = np.random.choice(list(colors.keys()))
        label_text = f"{num[y[i]] if nt[i] else y[i]}"
        is_prime = "prime" if check_prime(y[i]) else ""
        odd_even = "odd" if y[i] & 1 else "even"
        sentences = [
            f"{color_name} {label_text} {odd_even} {is_prime}",
            f"The {label_text} {color_name} {odd_even} {is_prime}",
            f"{color_name} image {label_text} {odd_even} {is_prime}",
        ]
        tx.append(np.random.choice(sentences))
        color = colors[color_name]
        x_color[i, x[i, :, :] == 1, 0] = color[0]
        x_color[i, x[i, :, :] == 1, 1] = color[1]
        x_color[i, x[i, :, :] == 1, 2] = color[2]

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    y_color = vectorizer.fit_transform(tx).toarray()

    d = vectorizer.get_feature_names_out()

    return x_color, y_color, tx, d


def dataset_mnist_color():
    data = MNIST().get()
    xtrain_orig, xtest_orig, ytrain_orig, ytest_orig = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
    )

    xtrain, ytrain, txtrain, d1 = add_color_mnist(xtrain_orig, ytrain_orig)
    xtest, ytest, txtest, d2 = add_color_mnist(xtest_orig, ytest_orig)
    assert (d1 == d2).all(), f"d1 and d2 are not same. \n{d1}\n{d2}"

    original = (xtrain_orig, xtest_orig, ytrain_orig, ytest_orig)
    return original, xtrain, ytrain, txtrain, xtest, ytest, txtest, d1


def arr_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b != 0)


def metrics(true, pred):
    land = np.logical_and(true, pred)
    lor = np.logical_or(true, pred)
    lxor = np.logical_xor(true, pred)  # symmetric diff
    n_correct_labels = np.sum(land, axis=1)
    total_active_labels = np.sum(lor, axis=1)
    n_miss_preds = np.sum(lxor, axis=1)

    acc = np.mean(arr_div(n_correct_labels, total_active_labels))
    pre = np.mean(arr_div(n_correct_labels, np.sum(pred, axis=1)))
    rec = np.mean(arr_div(n_correct_labels, np.sum(true, axis=1)))
    f1s = 2 * pre * rec / (pre + rec)
    hml = np.sum(n_miss_preds) / (true.shape[0] * true.shape[1])

    return {
        "Hamming loss": hml,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1 score": f1s,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clauses", default=2000, type=int)
    parser.add_argument("--T", default=3125, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--q", default=-1, type=float)
    parser.add_argument("--type_ratio", default=1.0, type=float)
    parser.add_argument("--platform", default="GPU", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--patch", default=10, type=int)
    args = parser.parse_args()

    params = dict(
        number_of_clauses=args.clauses,
        T=args.T,
        s=args.s,
        q=args.q,
        type_i_ii_ratio=args.type_ratio,
        patch_dim=(args.patch, args.patch),
        platform=args.platform,
        seed=10,
    )

    original, xtrain, ytrain, txtrain, xtest, ytest, txtest, label_names = dataset_mnist_color()

    tm = TMCoalesceMultiOuputClassifier(**params)

    print("Training with params: ")
    pprint(params)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        tm.fit(xtrain, ytrain, progress_bar=True)
        pred = tm.predict(xtest, progress_bar=True)

        met = metrics(ytest, pred)
        rep = classification_report(ytest, pred, target_names=label_names)

        pprint(met)
        print(rep)
        print("------------------------------")
