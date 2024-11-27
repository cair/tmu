import argparse
from pprint import pprint

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tmu.experimental.models.multioutput_classifier import TMCoalesceMultiOuputClassifier


def dataset_fmnist_ch(ch=8):
    X, y = fetch_openml(
        "Fashion-MNIST",
        version=1,
        return_X_y=True,
        as_frame=False,
    )

    xtrain_orig, xtest_orig, ytrain_orig, ytest_orig = train_test_split(X, y, random_state=0, test_size=10000)
    ytrain_orig = np.array(ytrain_orig, dtype=int)
    ytest_orig = np.array(ytest_orig, dtype=int)

    xtrain = np.array(xtrain_orig).reshape(-1, 28, 28)
    xtest = np.array(xtest_orig).reshape(-1, 28, 28)

    out = np.zeros((*xtrain.shape, ch))
    for j in range(ch):
        t1 = (j + 1) * 255 / (ch + 1)
        t2 = (j + 2) * 255 / (ch + 1)
        out[:, :, :, j] = np.logical_and(xtrain >= t1, xtrain < t2) & 1
    xtrain = np.array(out)

    out = np.zeros((*xtest.shape, ch))
    for j in range(ch):
        t1 = (j + 1) * 255 / (ch + 1)
        t2 = (j + 2) * 255 / (ch + 1)
        out[:, :, :, j] = np.logical_and(xtest >= t1, xtest < t2) & 1
    xtest = np.array(out)

    ytrain = np.zeros((ytrain_orig.shape[0], 10), dtype=int)
    for i in range(ytrain_orig.shape[0]):
        ytrain[i, ytrain_orig[i]] = 1
    ytest = np.zeros((ytest_orig.shape[0], 10), dtype=int)
    for i in range(ytest_orig.shape[0]):
        ytest[i, ytest_orig[i]] = 1

    label_names = [
        "tshirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankleboot",
    ]
    original = (xtrain_orig, ytrain_orig, xtest_orig, ytest_orig)
    return original, xtrain, ytrain, xtest, ytest, label_names


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

    original, xtrain, ytrain, xtest, ytest, label_names = dataset_fmnist_ch()

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
