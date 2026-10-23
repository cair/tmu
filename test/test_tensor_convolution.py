import unittest

import numpy as np

from tmu.models.classification.vanilla_classifier import TMClassifier


def _tensor_device_available():
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() < 1:
            return False
        return tuple(cuda.Device(0).compute_capability()) >= (8, 0)
    except Exception:
        return False


@unittest.skipUnless(_tensor_device_available(), "Tensor Core GPU is not available")
class TensorConvolutionTest(unittest.TestCase):
    def _images(self):
        images = np.zeros((3, 4, 4), dtype=np.uint32)
        images[0, 0, 0] = 1
        images[1, :, :] = 1
        images[2, 1:3, 1:3] = 1
        labels = np.array([0, 0, 1], dtype=np.uint32)
        return images, labels

    def test_device_encoding_matches_tmu_tools(self):
        import pycuda.driver as cuda

        images, labels = self._images()
        config = dict(
            number_of_clauses=4,
            T=8,
            s=2.0,
            patch_dim=(2, 2),
            weighted_clauses=False,
            seed=1,
        )
        cpu = TMClassifier(platform="CPU", **config)
        tensor = TMClassifier(platform="tensor", **config)
        cpu.init(images, labels)
        tensor.init(images, labels)
        expected = cpu.clause_banks[0].prepare_X(images)
        device = tensor.clause_banks[0].stage_images(images)
        actual = np.empty_like(expected)
        cuda.memcpy_dtoh(actual, device)
        np.testing.assert_array_equal(actual, expected)

    def test_patch_outputs_match_the_cpu_bank(self):
        images, labels = self._images()
        config = dict(
            number_of_clauses=4,
            T=8,
            s=2.0,
            patch_dim=(2, 2),
            weighted_clauses=False,
            seed=1,
        )
        cpu = TMClassifier(platform="CPU", **config)
        tensor = TMClassifier(platform="tensor", **config)
        cpu.init(images, labels)
        tensor.init(images, labels)
        for literal in (0, 1):
            cpu.clause_banks[0].set_ta_state(0, literal, 200)
            tensor.clause_banks[0].set_ta_state(0, literal, 200)
        cpu_encoded = cpu.clause_banks[0].prepare_X(images)
        tensor_encoded = tensor.clause_banks[0].prepare_X(images)
        for row in range(len(labels)):
            expected = cpu.clause_banks[0].calculate_clause_outputs_predict(cpu_encoded, row)
            actual = tensor.clause_banks[0].calculate_clause_outputs_predict(tensor_encoded, row)
            np.testing.assert_array_equal(actual, expected)

    def test_training_learns_a_corner_bit(self):
        rng = np.random.RandomState(0)
        images = rng.randint(0, 2, size=(60, 6, 6), dtype=np.uint32)
        labels = images[:, 0, 0].copy()
        model = TMClassifier(
            number_of_clauses=8,
            T=10,
            s=3.0,
            platform="tensor",
            patch_dim=(2, 2),
            weighted_clauses=True,
            batch_size=20,
            seed=1,
        )
        for _ in range(8):
            model.fit(images, labels, shuffle=True)
        accuracy = float(np.mean(model.predict(images) == labels))
        self.assertGreater(accuracy, 0.65)

    def test_short_literal_tail_matches_across_clause_panels(self):
        # 8x8x32, patch 2x2: 280 literals, 9 chunks. The last chunk is one word,
        # so the match has to add that word outside the 256-bit tensor tile.
        images = np.zeros((2, 8, 8, 32), dtype=np.uint32)
        images[1] = 1
        labels = np.array([0, 1], dtype=np.uint32)
        probe = TMClassifier(
            number_of_clauses=8,
            T=8,
            s=2.0,
            platform="tensor",
            patch_dim=(2, 2),
            weighted_clauses=False,
            seed=1,
        )
        probe.init(images, labels)
        runtime = probe.clause_banks[0]._runtime
        blocks_per_sm = max(1, int(runtime.max_threads_per_multiprocessor) // 128)
        target = int(runtime.multiprocessor_count) * blocks_per_sm
        clauses = target * 8
        config = dict(
            number_of_clauses=clauses,
            T=8,
            s=2.0,
            patch_dim=(2, 2),
            weighted_clauses=False,
            seed=1,
        )
        cpu = TMClassifier(platform="CPU", **config)
        tensor = TMClassifier(platform="tensor", **config)
        cpu.init(images, labels)
        tensor.init(images, labels)
        bank = tensor.clause_banks[0]
        self.assertEqual(int(bank.number_of_ta_chunks), 9)
        plan = runtime.conv_launch(clauses, 1, int(bank.number_of_patches), 9, len(images))
        self.assertEqual(plan.groups, clauses // 8)
        self.assertEqual(plan.skip_empty_words, 0)
        cpu.clause_banks[0].set_ta_state(0, 256, 200)
        bank.set_ta_state(0, 256, 200)
        encoded = np.ascontiguousarray(cpu.clause_banks[0].prepare_X(images))
        encoded = encoded.reshape(len(images), int(bank.number_of_patches), 9)
        encoded[:, :, 8] |= np.uint32(1)
        encoded = np.ascontiguousarray(encoded.reshape(len(images), -1))
        for row in range(len(images)):
            expected = cpu.clause_banks[0].calculate_clause_outputs_predict(encoded, row)
            actual = bank.calculate_clause_outputs_predict(encoded, row)
            np.testing.assert_array_equal(actual, expected)

    def test_channel_stack_matches_the_cpu_bank(self):
        # 4x4x48, patch 2x2: 392 literals, 13 words. The live-word kernel
        # has to keep a bit in the first word and a bit in word 12.
        images = np.zeros((2, 4, 4, 48), dtype=np.uint32)
        images[1, :, :, :8] = 1
        labels = np.array([0, 1], dtype=np.uint32)
        config = dict(
            number_of_clauses=16,
            T=8,
            s=2.0,
            patch_dim=(2, 2),
            weighted_clauses=False,
            seed=1,
        )
        cpu = TMClassifier(platform="CPU", **config)
        tensor = TMClassifier(platform="tensor", **config)
        cpu.init(images, labels)
        tensor.init(images, labels)
        bank = tensor.clause_banks[0]
        chunks = int(bank.number_of_ta_chunks)
        self.assertGreaterEqual(chunks, 12)
        plan = bank._runtime.conv_launch(
            16, 1, int(bank.number_of_patches), chunks, len(images))
        self.assertEqual(plan.skip_empty_words, 1)
        literals = int(cpu.clause_banks[0].number_of_literals)
        for clause in range(8):
            for literal in range(literals):
                cpu.clause_banks[0].set_ta_state(clause, literal, 0)
                bank.set_ta_state(clause, literal, 0)
        cpu.clause_banks[0].set_ta_state(0, 0, 200)
        cpu.clause_banks[0].set_ta_state(1, 384, 200)
        bank.set_ta_state(0, 0, 200)
        bank.set_ta_state(1, 384, 200)
        encoded = np.ascontiguousarray(cpu.clause_banks[0].prepare_X(images))
        encoded = encoded.reshape(len(images), int(bank.number_of_patches), chunks)
        encoded[:, :, 0] |= np.uint32(1)
        encoded[:, :, 12] |= np.uint32(1)
        encoded = np.ascontiguousarray(encoded.reshape(len(images), -1))
        for row in range(len(images)):
            expected = cpu.clause_banks[0].calculate_clause_outputs_predict(encoded, row)
            actual = bank.calculate_clause_outputs_predict(encoded, row)
            np.testing.assert_array_equal(actual, expected)

class ConvLaunchPlanTest(unittest.TestCase):
    def _runtime(self, sms, shared, threads_per_block=1024, threads_per_sm=1536):
        from tmu.clause_bank.clause_bank_tensor_core import build_tensor_core_runtime

        return build_tensor_core_runtime(
            (8, 9),
            {
                "MAX_THREADS_PER_BLOCK": threads_per_block,
                "MAX_THREADS_PER_MULTIPROCESSOR": threads_per_sm,
                "MAX_SHARED_MEMORY_PER_BLOCK": min(48 * 1024, shared),
                "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN": shared,
                "MULTIPROCESSOR_COUNT": sms,
                "MAX_GRID_DIM_Z": 65535,
            },
            12 * 1024**3,
        )

    def test_mnist_patch_uses_four_warps(self):
        plan = self._runtime(46, 100 * 1024).conv_launch(8000, 10, 361, 9, 1)
        self.assertEqual(plan.threads, 128)
        self.assertEqual(plan.groups, 1000)
        self.assertEqual(plan.skip_empty_words, 0)

    def test_a_deep_channel_stack_skips_empty_words(self):
        short = self._runtime(46, 100 * 1024).conv_launch(8000, 10, 361, 9, 1)
        cifar = self._runtime(46, 100 * 1024).conv_launch(8000, 10, 900, 18, 1)
        huge = self._runtime(46, 100 * 1024).conv_launch(8000, 10, 900, 300, 1)
        self.assertEqual(short.skip_empty_words, 0)
        self.assertEqual(cifar.skip_empty_words, 1)
        self.assertEqual(cifar.threads, 128)
        self.assertEqual(huge.skip_empty_words, 1)
        self.assertGreater(huge.shared_bytes, cifar.shared_bytes)

    def test_patch_bitset_follows_shared_memory(self):
        small = self._runtime(46, 8 * 1024)
        wide = self._runtime(46, 100 * 1024)
        self.assertEqual(small.max_conv_patches(), 8192)
        self.assertEqual(wide.max_conv_patches(), (100 * 1024 // 32) * 32)
        self.assertGreater(wide.max_conv_patches(), small.max_conv_patches())
        packed = small.conv_launch(8, 1, 100, 9, 1)
        roomy = wide.conv_launch(8, 1, 20000, 9, 1)
        self.assertEqual(packed.bit_words, (100 + 31) // 32)
        self.assertEqual(packed.shared_bytes, packed.bit_words * 8 * 4)
        self.assertLessEqual(packed.shared_bytes, 8 * 1024)
        self.assertEqual(roomy.bit_words, (20000 + 31) // 32)
        self.assertGreater(roomy.shared_bytes, 8 * 1024)
        self.assertLessEqual(roomy.shared_bytes, 100 * 1024)

    def test_live_index_stops_when_shared_memory_is_full(self):
        roomy = self._runtime(46, 100 * 1024).conv_launch(8000, 10, 900, 300, 1)
        tight = self._runtime(46, 1024).conv_launch(8000, 10, 900, 300, 1)
        self.assertEqual(roomy.skip_empty_words, 1)
        self.assertGreater(roomy.shared_bytes, roomy.bit_words * 8 * 4)
        self.assertLessEqual(roomy.shared_bytes, 100 * 1024)
        self.assertEqual(tight.skip_empty_words, 0)
        self.assertEqual(tight.shared_bytes, tight.bit_words * 8 * 4)
        self.assertLessEqual(tight.shared_bytes, 1024)

    def test_one_image_keeps_one_panel_per_block(self):
        from tmu.clause_bank.clause_bank_tensor_core import conv_image_cache

        runtime = self._runtime(46, 100 * 1024)
        launch = runtime.conv_launch(4000, 10, 361, 9, 1)
        cache = conv_image_cache(
            runtime.shared_memory_per_block,
            launch.threads,
            runtime.max_threads_per_multiprocessor,
            runtime.multiprocessor_count,
            launch.shared_bytes,
            launch.groups,
            361,
            9,
            10,
            1,
        )
        self.assertEqual(cache.x_words, 0)
        self.assertEqual(cache.panels_per, 1)
        self.assertEqual(cache.groups, launch.groups)
        self.assertEqual(cache.shared_bytes, launch.shared_bytes)

    def test_a_small_grid_keeps_one_panel_per_block(self):
        from tmu.clause_bank.clause_bank_tensor_core import conv_image_cache

        runtime = self._runtime(46, 100 * 1024)
        launch = runtime.conv_launch(8, 2, 25, 1, 1)
        cache = conv_image_cache(
            runtime.shared_memory_per_block,
            launch.threads,
            runtime.max_threads_per_multiprocessor,
            runtime.multiprocessor_count,
            launch.shared_bytes,
            launch.groups,
            25,
            1,
            2,
            1,
        )
        self.assertEqual(cache.panels_per, 1)
        self.assertEqual(cache.groups, launch.groups)

    def test_image_that_does_not_fit_is_not_cached(self):
        from tmu.clause_bank.clause_bank_tensor_core import conv_image_cache

        runtime = self._runtime(46, 48 * 1024)
        launch = runtime.conv_launch(800, 10, 20000, 64, 1)
        cache = conv_image_cache(
            runtime.shared_memory_per_block,
            launch.threads,
            runtime.max_threads_per_multiprocessor,
            runtime.multiprocessor_count,
            launch.shared_bytes,
            launch.groups,
            20000,
            64,
            10,
            1,
        )
        self.assertEqual(cache.x_words, 0)
        self.assertEqual(cache.panels_per, 1)
        self.assertEqual(cache.shared_bytes, launch.shared_bytes)

    def test_short_rows_stay_dense_on_a_large_gpu(self):
        runtime = self._runtime(46, 100 * 1024)
        self.assertEqual(runtime.conv_launch(8, 1, 100, 11, 1).skip_empty_words, 0)
        twelve = runtime.conv_launch(8, 1, 100, 12, 1)
        self.assertEqual(twelve.skip_empty_words, 1)
        self.assertGreater(twelve.shared_bytes, twelve.bit_words * 8 * 4)

    def test_patch_width_and_device_both_change_the_block(self):
        runtime = self._runtime(46, 100 * 1024)
        small = runtime.conv_launch(8, 2, 25, 1, 1)
        wide = runtime.conv_launch(8000, 10, 900, 18, 1)
        capped = self._runtime(46, 48 * 1024, threads_per_block=64).conv_launch(8000, 10, 900, 18, 1)
        narrow = self._runtime(4, 48 * 1024).conv_launch(8000, 10, 361, 9, 1)
        self.assertEqual(small.threads, 64)
        self.assertEqual(wide.threads, 128)
        self.assertEqual(capped.threads, 64)
        self.assertEqual(narrow.threads, 128)
        self.assertEqual(narrow.groups, 1000)


if __name__ == "__main__":
    unittest.main()
