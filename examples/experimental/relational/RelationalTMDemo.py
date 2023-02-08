import numpy as np
import keras
from time import time

from tmu.models.relational.vanilla_relational import TMRelational

X = [
		((("Parent", "Mary", "Bob"), True), (("Parent", "Bob", "Peter"), True), (("Ancestor", "Mary", "Bob"), True), (("Ancestor", "Bob", "Peter"), True), (("Ancestor", "Mary", "Peter"), True)),
		((("Parent", "Ida", "Chris"), True), (("Parent", "Chris", "Ann"), True), (("Ancestor", "Ida", "Chris"), True), (("Ancestor", "Chris", "Ann"), True), (("Ancestor", "Ida", "Ann"), True))
	]

output_active = [("Ancestor", "Ida", "Chris"), ("Ancestor", "Chris", "Ann"), ("Ancestor", "Ida", "Ann"), ("Ancestor", "Mary", "Bob"), ("Ancestor", "Bob", "Peter"), ("Ancestor", "Mary", "Peter")]

clauses = 20
T = 40
s = 5.0

print("Number of clauses:", clauses)

tm = TMRelational(clauses, T, s, output_active, max_included_literals=3, feature_negation=False, platform='CPU', output_balancing=True)

print("\nAccuracy Over 40 Epochs:")
for e in range(40):
	tm.fit(X)
	print(e)