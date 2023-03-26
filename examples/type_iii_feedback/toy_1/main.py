import os
import dotenv
from pgmpy.sampling import BayesianModelSampling
import argparse
import optuna_distributed
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import optuna
from tqdm import tqdm
from examples.type_iii_feedback.toy_1.objectives import type_iii_optimizer_v1, type_iii_optimizer_v2
import models
from examples.type_iii_feedback.utils import draw_bayesian_network, get_boundary, DummyTrial

if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clauses", default=100, type=int)
    parser.add_argument("--T", default=10, type=int)
    parser.add_argument("--s", default=50.0, type=float)
    parser.add_argument("--max_included-literals", default=32, type=int)
    parser.add_argument("--device", default="CPU", type=str)
    parser.add_argument("-d", default=200, type=int)
    parser.add_argument("--weighted-clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-state-bits-ta", default=8, type=int)
    parser.add_argument("--number-of-state-bits-ind", default=8, type=int)
    parser.add_argument("--iii", default=True, type=bool)
    parser.add_argument("--mysql-user", default=os.getenv("MYSQL_USER", default=None))
    parser.add_argument("--mysql-pass", default=os.getenv("MYSQL_PASS", default=None))
    parser.add_argument("--mysql-host", default=os.getenv("MYSQL_HOST", default=None))
    parser.add_argument("--mysql-db", default=os.getenv("MYSQL_DB", default=None))
    parser.add_argument("--optuna-storage", default="sqlite", choices=["sqlite", "mysql"])
    args = parser.parse_args("")

    bn_model, bn_target = models.toy_bn()

    G = draw_bayesian_network(bn_model)
    labels = {str(x): str(x) for i, x in enumerate(bn_model.nodes)}
    target_boundary = get_boundary(bn_model, bn_target)
    sampler = BayesianModelSampling(bn_model)

    # Sample data from the model
    data = sampler.forward_sample(size=5000)

    TARGET = bn_target if isinstance(bn_target, list) else [bn_target]
    FEATURES = list(bn_model.nodes)
    FEATURES = [x for x in FEATURES if x not in TARGET]

    def run_experiment(trial, optimizer, dry=False):

        import logging
        from tmu.models.classification.vanilla_classifier import TMClassifier

        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.WARNING)

        X = data[FEATURES].values
        y = data[TARGET].values.flatten()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

        num_clauses = trial.suggest_int("num_clauses", 10, 300)  # clauses
        num_clauses = num_clauses if num_clauses % 2 == 0 else num_clauses + 1
        d = trial.suggest_float("d", 500, 300.0)  # s
        sensitivity = trial.suggest_float("sensitivity", 1.0, 300.0)  # s
        threshold = trial.suggest_int("threshold", 1, 200)  # T
        max_included_literals = args.max_included_literals
        number_of_state_bits_ta = trial.suggest_int("number_of_state_bits_ta", 1, 64)  # args.number_of_state_bits_ta
        number_of_state_bits_ind = trial.suggest_int("number_of_state_bits_ind", 1, 64)  # args.number_of_state_bits_ind
        platform = trial.suggest_categorical("platform", ["CPU"])
        weighted_clauses = True
        type_iii_feedback = True  # trial.suggest_categorical("type-iii", [True, False])

        tm = TMClassifier(
            number_of_clauses=num_clauses,
            T=threshold,
            s=sensitivity,
            d=d,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            number_of_state_bits_ind=number_of_state_bits_ind,
            platform=platform,
            weighted_clauses=weighted_clauses,
            type_iii_feedback=type_iii_feedback,
            ta_state_ind_init_value=5000
        )

        if dry:
            for x in tqdm(range(args.epochs)):
                tm.fit(X_train, y_train)
                y_pred = tm.predict(X_test)
        else:
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
    score = run_experiment(DummyTrial(), lambda m: (type_iii_optimizer_v1(m, FEATURES, target_boundary), type_iii_optimizer_v2(m, FEATURES, target_boundary)), dry=True)
    print(score)

    study = optuna_distributed.from_study(optuna.create_study(
        storage="sqlite:///db.sqlite3" if args.optuna_storage == "sqlite" else f"mysql+pymysql://{args.mysql_user}:{args.mysql_pass}@{args.mysql_host}/{args.mysql_db}",
        study_name=f"type-iii-TM-{int(time.time())}",
        directions=["maximize", "maximize"]
    ), client=None)

    study.optimize(
        lambda trial: run_experiment(trial, lambda m: (
            type_iii_optimizer_v1(m, FEATURES, target_boundary),
            type_iii_optimizer_v2(m, FEATURES, target_boundary))
                                     ),
        n_trials=100000,
        show_progress_bar=False,
        n_jobs=os.cpu_count() * 3
    )
