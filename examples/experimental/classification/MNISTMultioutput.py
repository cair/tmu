import argparse
from pprint import pprint

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss, precision_score, recall_score
from tmu.data import MNIST
from tmu.experimental.models.multioutput_classifier import TMCoalesceMultiOuputClassifier


def dataset_mnist():
    data = MNIST().get()
    xtrain_orig, xtest_orig, ytrain_orig, ytest_orig = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
    )

    xtrain = xtrain_orig.reshape(-1, 28, 28)
    xtest = xtest_orig.reshape(-1, 28, 28)

    ytrain = np.zeros((ytrain_orig.shape[0], 10), dtype=int)
    for i in range(ytrain_orig.shape[0]):
        ytrain[i, ytrain_orig[i]] = 1
    ytest = np.zeros((ytest_orig.shape[0], 10), dtype=int)
    for i in range(ytest_orig.shape[0]):
        ytest[i, ytest_orig[i]] = 1

    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    original = (xtrain_orig, ytrain_orig, xtest_orig, ytest_orig)
    return original, xtrain, ytrain, xtest, ytest, label_names

def metrics(true, pred):
    met = {
        "Subset accuracy": accuracy_score(true, pred),
        "Hamming loss": hamming_loss(true, pred),
        "F1 score": f1_score(true, pred, average="weighted"),
        "Precision": precision_score(true, pred, average="weighted"),
        "Recall": recall_score(true, pred, average="weighted"),
    }
    return met

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

    original, xtrain, ytrain, xtest, ytest, label_names = dataset_mnist()

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
