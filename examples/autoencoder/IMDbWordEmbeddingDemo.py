import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from tmu.data import IMDB
from tmu.tools import BenchmarkTimer
import logging
from tmu.util.cuda_profiler import CudaProfiler
import os
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



import argparse

def main():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--clause_weight_threshold', type=int, default=0, help='Clause weight threshold')
	parser.add_argument('--number_of_examples', type=int, default=2000, help='Number of examples')
	parser.add_argument('--accumulation', type=int, default=25, help='Accumulation')
	parser.add_argument('--factor', type=int, default=4, help='Factor')
	parser.add_argument('--clauses', type=int, default=80, help='Clauses')  # factor*20
	parser.add_argument('--T', type=int, default=160, help='T') # factor*40
	parser.add_argument('--s', type=float, default=5.0, help='S')
	parser.add_argument('--NUM_WORDS', type=int, default=10000, help='Number of words')
	parser.add_argument('--INDEX_FROM', type=int, default=2, help='Index from')
	parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
	parser.add_argument('--device', type=str, default="CUDA", help='which device to use')
	parser.add_argument("--number_of_output_words", type=int, default=20, help="how many words")
 
	args = parser.parse_args()

	# load IMDB dataset
	dataloader = IMDB(num_words=args.NUM_WORDS, index_from=args.INDEX_FROM)
	dataset = dataloader.get()
	train_x = dataset["x_train"]
	train_y = dataset["y_train"]
	test_x = dataset["x_test"]
	test_y = dataset["y_test"]

	# get word to indice and id_to_word mappings
	word_to_id = dataloader.get_word_index()
	word_to_id = {k:(v+args.INDEX_FROM) for k,v in word_to_id.items()}
	word_to_id["<PAD>"] = 0
	word_to_id["<START>"] = 1
	word_to_id["<UNK>"] = 2
	id_to_word = {value:key for key,value in word_to_id.items()}

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
	number_of_features = vectorizer_X.get_feature_names_out().shape[0]
 
	target_words = []
	for word in feature_names:
		word_id = vectorizer_X.vocabulary_[word]
		target_words.append(word)
		if len(target_words) == args.number_of_output_words:
			break

	X_test = vectorizer_X.transform(testing_documents)

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
		max_included_literals=3,
		accumulation=args.accumulation, 
		feature_negation=False, 
		platform=args.device, 
		output_balancing=True
     )


	benchmark_train = BenchmarkTimer()
	benchmark_test = BenchmarkTimer()
	benchmark_total = BenchmarkTimer()
 
	for e in range(args.epochs):
		with benchmark_total:
			with benchmark_train:
				tm.fit(X_train, number_of_examples=args.number_of_examples)


			_LOGGER.info("\nEpoch #%d\n" % (e+1))

			_LOGGER.info("Calculating precision\n")
			precision = []
			for i in range(len(target_words)):
				precision.append(tm.clause_precision(i, True, X_train, number_of_examples=500))

			_LOGGER.info("Calculating recall\n")
			recall = []
			for i in range(len(target_words)):
				recall.append(tm.clause_recall(i, True, X_train, number_of_examples=500))

			_LOGGER.info("Clauses\n")

			for j in range(args.clauses):
				_LOGGER.info("Clause #%d " % (j))
				for i in range(len(target_words)):
					_LOGGER.info("%s:W%d:P%.2f:R%.2f " % (target_words[i], tm.get_weight(i, j), precision[i][j], recall[i][j]))

				l = []
				for k in range(tm.clause_bank.number_of_literals):
					if tm.get_ta_action(j, k) == 1:
						if k < tm.clause_bank.number_of_features:
							l.append("%s(%d)" % (feature_names[k], tm.clause_bank.get_ta_state(j, k)))
						else:
							l.append("¬%s(%d)" % (feature_names[k-tm.clause_bank.number_of_features], tm.clause_bank.get_ta_state(j, k)))
				_LOGGER.info(" ∧ ".join(l))

			profile = np.empty((len(target_words), args.clauses))
			for i in range(len(target_words)):
				weights = tm.get_weights(i)
				profile[i,:] = np.where(weights >= args.clause_weight_threshold, weights, 0)

			similarity = cosine_similarity(profile)

			_LOGGER.info("\nWord Similarity\n")

			for i in range(len(target_words)):
				_LOGGER.info(target_words[i])
				sorted_index = np.argsort(-1 * similarity[i,:])
				for j in range(1, len(target_words)):
					_LOGGER.info("%s(%.2f) " % (target_words[sorted_index[j]], similarity[i,sorted_index[j]]))
				_LOGGER.info("")

		if args.device == "CUDA":
			CudaProfiler().print_timings(benchmark=benchmark_total)



if __name__ == "__main__":
    
    main()