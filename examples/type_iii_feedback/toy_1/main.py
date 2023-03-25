from pgmpy.sampling import BayesianModelSampling
import argparse
import optuna_distributed
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import optuna
import numpy as np
from termcolor import colored
from type_iii_feedback.utils import get_boundary, draw_network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=100, type=int)
    parser.add_argument("--T", default=10, type=int)
    parser.add_argument("--s", default=50.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="CPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--number-of-state-bits-ta", default=8, type=int)
    parser.add_argument("--number-of-state-bits-ind", default=8, type=int)
    parser.add_argument("--iii", default=True, type=bool)
    args = parser.parse_args("")

    G = draw_network(model)
    labels = {str(x): str(x) for i, x in enumerate(model.nodes)}
    target_boundary = get_boundary("Xray", labels, labels, G)

    sampler = BayesianModelSampling(model)

    # Sample data from the model
    data = sampler.forward_sample(size=500)

    FEATURES = list(model.nodes)
    TARGET = ["Xray"]


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


    def type_iii_optimizer_base(model):
        # Let all clauses be active
        clause_active = np.ones(model.number_of_clauses, dtype=np.uint32)

        # Calculate all clause literals for all classes
        clauses_inclusions = np.vstack(
            [x.calculate_literal_clause_frequency_individual_clause(clause_active) for x in model.clause_banks]
        )

        return clauses_inclusions


    def type_iii_optimizer_v1(model):
        """
        This optimizer will only count the ratio of which blanket variables are present
        :param model:
        :return:
        """
        clauses_inclusions = type_iii_optimizer_base(model)

        blanket_ratio = np.zeros(shape=(len(clauses_inclusions, )))
        for i, clause in enumerate(clauses_inclusions):
            rule_data = evaluate_rule(clause, labels=FEATURES, blanket=target_boundary)
            blanket_ratio[i] = rule_data["blanket_ratio"]

        return blanket_ratio.mean()


    def type_iii_optimizer_v2(model):
        """
        This optimizer, will maximize the number of rules that have the full blanket present (at least)
        :param model:
        :return:
        """
        clauses_inclusions = type_iii_optimizer_base(model)

        blanket_ratio = np.zeros(shape=(len(clauses_inclusions, )))
        for i, clause in enumerate(clauses_inclusions):
            rule_data = evaluate_rule(clause, labels=FEATURES, blanket=target_boundary)
            blanket_ratio[i] = int(rule_data["blanket_present"])

        return blanket_ratio.sum() / len(blanket_ratio)



    def run_experiment(trial, optimizer):

        import logging
        from tmu.models.classification.vanilla_classifier import TMClassifier

        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.WARNING)

        X = data[FEATURES].values
        y = data[TARGET].values.flatten()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

        num_clauses = trial.suggest_int("num_clauses", 10, 50)  # clauses
        num_clauses = num_clauses if num_clauses % 2 == 0 else num_clauses + 1
        sensitivity = trial.suggest_float("sensitivity", 1.0, 300.0)  # s
        threshold = trial.suggest_int("threshold", 1, 100)  # T
        max_included_literals = args.max_included_literals
        number_of_state_bits_ta = trial.suggest_int("number_of_state_bits_ta", 1, 64)  # args.number_of_state_bits_ta
        number_of_state_bits_ind = trial.suggest_int("number_of_state_bits_ind", 1, 64)  # args.number_of_state_bits_ind
        platform = "CPU"
        weighted_clauses = True
        type_iii_feedback = True  # trial.suggest_categorical("type-iii", [True, False])

        tm = TMClassifier(
            number_of_clauses=num_clauses,
            T=threshold,
            s=sensitivity,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            number_of_state_bits_ind=number_of_state_bits_ind,
            platform=platform,
            weighted_clauses=weighted_clauses,
            type_iii_feedback=type_iii_feedback
        )

        for x in range(args.epochs):
            tm.fit(X_train, y_train)
            y_pred = tm.predict(X_test)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_pred, y_pred)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("f-score", fscore)
        trial.set_user_attr("support", support)
        trial.set_user_attr("accuracy", accuracy)

        return optimizer(tm)


    # Dry run
    score = run_experiment(DummyTrial(), lambda m: (type_iii_optimizer_v1(m), type_iii_optimizer_v2(m)))
    print(score)

    study = optuna_distributed.from_study(optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=f"type-iii-TM-{int(time.time())}",
        directions=["maximize", "maximize"]
    ), client=None)

    study.optimize(
        lambda trial: run_experiment(trial, lambda m: (type_iii_optimizer_v1(m), type_iii_optimizer_v2(m))),
        n_trials=100000,
        show_progress_bar=False,
        n_jobs=10
    )
