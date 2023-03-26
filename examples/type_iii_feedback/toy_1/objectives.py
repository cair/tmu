import numpy as np
from termcolor import colored


def type_iii_optimizer_base(model):
    # Calculate all clause literals for all classes
    clauses_inclusions = np.vstack(
        [model.clause_banks[x].get_literals() for x in model.clause_banks]
    )
    return clauses_inclusions


def evaluate_rule(clause, labels=None, blanket=None):
    ret = dict(
        summary="N/A",
        blanket_ratio=0,
        blanket_present=False
    )

    # Add negated label
    if labels is None:
        labels_with_negated = [f"X{i}" for i in range(len(clause / 2))] + [f"¬X{i}" for i in range(len(clause / 2))]
    else:
        labels_with_negated = labels + [f"¬{l}" for l in labels]

    # Add negated blanket
    if blanket:
        blanket_w_neg = blanket + [f"¬{l}" for l in blanket]
    else:
        blanket_w_neg = []

    # colored(label, "red") if label in blanket_w_neg else

    # Create list of all included literals
    inclusions = set([label for label, literal in zip(labels_with_negated, clause) if literal >= 1])

    if len(inclusions) == 0:
        return ret

    # Remove those in the blanket
    inclusions_without_blanket = inclusions - set(blanket_w_neg)

    # Compute the size difference between
    incl_diff = len(inclusions) - len(inclusions_without_blanket)

    # Blanket must be present if the size of the blanket is in diff to inclusions without blanket
    blanket_present = incl_diff >= len(blanket)
    blanket_ratio = incl_diff / len(inclusions)

    # generate final string
    out = ' ^ '.join([colored(x, "red") if x in blanket_w_neg else x for x in inclusions])

    ret["summary"] = out
    ret["blanket_ratio"] = blanket_ratio
    ret["blanket_present"] = blanket_present

    return ret


def type_iii_optimizer_v1(model, X, y):
    """
    This optimizer will only count the ratio of which blanket variables are present
    :param model:
    :return:
    """
    clauses_inclusions = type_iii_optimizer_base(model)

    blanket_ratio = np.zeros(shape=(len(clauses_inclusions, )))

    for i, clause in enumerate(clauses_inclusions):
        rule_data = evaluate_rule(clause, labels=X, blanket=y)
        blanket_ratio[i] = rule_data["blanket_ratio"]

    return blanket_ratio.mean()


def type_iii_optimizer_v2(model, X, y):
    """
    This optimizer, will maximize the number of rules that have the full blanket present (at least)
    :param model:
    :return:
    """
    clauses_inclusions = type_iii_optimizer_base(model)

    blanket_ratio = np.zeros(shape=(len(clauses_inclusions, )))

    for i, clause in enumerate(clauses_inclusions):
        rule_data = evaluate_rule(clause, labels=X, blanket=y)
        blanket_ratio[i] = int(rule_data["blanket_present"])

    return blanket_ratio.sum() / len(blanket_ratio)


def type_iii_optimizer_v3(model, X, y):
    """
    This optimizer will only count fill blanket clauses
    :param model:
    :return:
    """
    clauses_inclusions = type_iii_optimizer_base(model)

    blanket_clauses = 0
    for i, clause in enumerate(clauses_inclusions):
        rule_data = evaluate_rule(clause, labels=X, blanket=y)
        if rule_data["blanket_ratio"] >= 1.0:
            blanket_clauses += 1

    return blanket_clauses / len(clauses_inclusions)
