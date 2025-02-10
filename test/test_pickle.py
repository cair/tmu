import unittest
import pickle
import numpy as np
from tmu.clause_bank.clause_bank_cuda import ClauseBankCUDA
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier
import tqdm

class TestClauseBankPickle(unittest.TestCase):
    def test_pickle(self):
        # Create minimal test data - 100 samples, 10x10 image
        X = np.random.randint(0, 2, size=(100, 10, 10))

        # Initialize ClauseBankCUDA with parameters matching _build_gpu_bank
        # patch_dim must be <= input dimensions to ensure positive number_of_patches
        # number_of_patches = (dim[0] - patch_dim[0] + 1) * (dim[1] - patch_dim[1] + 1)
        model = ClauseBankCUDA(
            X_shape=X.shape,
            s=10.0,
            boost_true_positive_feedback=1,
            reuse_random_feedback=True,
            number_of_clauses=10,
            number_of_state_bits_ta=8,
            patch_dim=(5, 5),  # 5x5 patches on 10x10 input gives 36 patches
            type_ia_ii_feedback_ratio=1.0,
            max_included_literals=10,
            seed=42
        )

        # Serialize the model using pickle (the device branch will be excluded).
        pickled_model = pickle.dumps(model)

        # Deserialize the model; the __setstate__ will reinitialize the device branch.
        model_unpickled = pickle.loads(pickled_model)

        # Check that the host state is exactly preserved.
        self.assertEqual(model.host.clause_bank.shape, model_unpickled.host.clause_bank.shape)
        self.assertEqual(model.get_ta_state(0, 0), model_unpickled.get_ta_state(0, 0))

        # Verify that the device branch has been reinitialized.
        if model_unpickled.device is not None:
            model_unpickled.device.synchronize_clause_bank()
            self.assertTrue(hasattr(model_unpickled.device, 'clause_bank_gpu'))
        else:
            self.skipTest("CUDA is not available in the unpickled model.")

class TestMNISTPickle(unittest.TestCase):
    def test_mnist_pickle(self):
        # Load MNIST data
        data = MNIST().get()

        # limit the data to 1000 samples
        data["x_train"] = data["x_train"][:1000]
        data["y_train"] = data["y_train"][:1000]
        data["x_test"] = data["x_test"][:1000]
        data["y_test"] = data["y_test"][:1000]
        
        # Initialize classifier with smaller parameters for faster testing
        tm = TMClassifier(
            number_of_clauses=100,  # Reduced for faster testing
            T=50,                   # Reduced for faster testing
            s=10.0,
            max_included_literals=32,
            platform='CUDA',         # Using CPU to ensure consistent behavior
            weighted_clauses=True,
            seed=42
        )

        # Train for 5 epochs
        for _ in tqdm.tqdm(range(5)):
            tm.fit(
                data["x_train"].astype(np.uint32),
                data["y_train"].astype(np.uint32)
            )

        # Get predictions before serialization
        original_predictions = tm.predict(data["x_test"])
        original_accuracy = 100 * (original_predictions == data["y_test"]).mean()

        # Serialize the model
        pickled_model = pickle.dumps(tm)

        # Deserialize the model
        tm_unpickled = pickle.loads(pickled_model)

        # Get predictions after deserialization
        new_predictions = tm_unpickled.predict(data["x_test"])
        new_accuracy = 100 * (new_predictions == data["y_test"]).mean()


        print(original_accuracy, new_accuracy)

        # Verify predictions are exactly the same
        np.testing.assert_array_equal(
           original_predictions, 
           new_predictions, 
           err_msg="Predictions differ after pickle/unpickle"
        )

        # Verify accuracies match
        self.assertEqual(
           original_accuracy,
           new_accuracy,
           msg="Accuracy changed after pickle/unpickle"
        )

        # Additional structural checks
        self.assertEqual(
            tm.number_of_clauses,
            tm_unpickled.number_of_clauses,
            msg="number_of_clauses changed after pickle/unpickle"
        )
        
        # Check all TA states for each class
        for class_id in range(len(tm.clause_banks)):
            original_bank = tm.clause_banks[class_id]
            unpickled_bank = tm_unpickled.clause_banks[class_id]
            
            # Check all clauses
            for clause in range(original_bank.number_of_clauses):
                # Check all TAs in each clause
                for ta in range(original_bank.number_of_literals):
                    original_state = original_bank.get_ta_state(clause, ta)
                    unpickled_state = unpickled_bank.get_ta_state(clause, ta)
                    self.assertEqual(
                        original_state,
                        unpickled_state,
                        msg=f"TA state mismatch at class={class_id}, clause={clause}, ta={ta}"
                    )
            
            # Also verify clause bank shapes match
            self.assertEqual(
                original_bank.host.clause_bank.shape,
                unpickled_bank.host.clause_bank.shape,
                msg=f"Clause bank shape mismatch for class {class_id}"
            )
            
            # And verify the entire clause banks are identical
            np.testing.assert_array_equal(
                original_bank.host.clause_bank,
                unpickled_bank.host.clause_bank,
                err_msg=f"Clause bank content mismatch for class {class_id}"
            )

if __name__ == '__main__':
    unittest.main() 