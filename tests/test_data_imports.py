import unittest


class DataImportsTest(unittest.TestCase):
    def test_kaggle_resolver_imports(self):
        from nesy_gen.data.kaggle import DatasetPaths, print_dataset_paths, resolve_kaggle_dataset

        self.assertTrue(callable(resolve_kaggle_dataset))
        self.assertTrue(callable(print_dataset_paths))
        self.assertEqual(DatasetPaths.__name__, "DatasetPaths")


if __name__ == "__main__":
    unittest.main()

