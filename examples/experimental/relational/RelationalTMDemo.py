import numpy as np
import keras
from time import time

from tmu.models.relational.vanilla_relational import TMRelational

X = [
		(("Parent", "Mary", "Bob"), ("Parent", "Bob", "Peter"), ("Ancestor", "Mary", "Bob"), ("Ancestor", "Bob", "Peter"), ("Ancestor", "Mary", "Peter")),
		(("Parent", "Ida", "Chris"), ("Parent", "Chris", "Ann"), ("Ancestor", "Ida", "Chris"), ("Ancestor", "Chris", "Ann"), ("Ancestor", "Ida", "Ann"))
	]

clauses = 20
T = 40
s = 5.0

print("Number of clauses:", clauses)

tm = TMRelational(clauses, T, s, output_active, max_included_literals=3, feature_negation=False, platform='CPU', output_balancing=True)

tm.propositionalize_relations(X)
