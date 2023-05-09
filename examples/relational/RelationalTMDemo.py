import argparse

from tmu.models.relational.vanilla_relational import TMRelational
import logging

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=20, type=int)
    parser.add_argument("--T", default=40, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--max-included-literals", default=3, type=int)
    parser.add_argument("--feature-negation", default=False, type=bool)
    parser.add_argument("--output-balancing", default=True, type=bool)
    parser.add_argument("--device", default="CPU", type=str)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()

    X = [
        (
            (("Parent", "Mary", "Bob"), True),
            (("Parent", "Bob", "Peter"), True),
            (("Ancestor", "Mary", "Bob"), True),
            (("Ancestor", "Bob", "Peter"), True),
            (("Ancestor", "Mary", "Peter"), True)),
        (
            (("Parent", "Ida", "Chris"), True),
            (("Parent", "Chris", "Ann"), True),
            (("Ancestor", "Ida", "Chris"), True),
            (("Ancestor", "Chris", "Ann"), True),
            (("Ancestor", "Ida", "Ann"), True)
        )
    ]

    output_active = [
        ("Ancestor", "Ida", "Chris"),
        ("Ancestor", "Chris", "Ann"),
        ("Ancestor", "Ida", "Ann"),
        ("Ancestor", "Mary", "Bob"),
        ("Ancestor", "Bob", "Peter"),
        ("Ancestor", "Mary", "Peter")
    ]

    tm = TMRelational(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        output_active_facts=output_active,
        max_included_literals=args.max_included_literals,
        feature_negation=args.feature_negation,
        platform=args.device,
        output_balancing=args.output_balancing
    )

    _LOGGER.info(f"Accuracy over {args.epochs} epochs.")
    for epoch in range(args.epochs):
        tm.fit(X)
        _LOGGER.info(f"Epoch: {epoch}")
