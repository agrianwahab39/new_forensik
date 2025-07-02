import unittest
import numpy as np

from classification import prepare_feature_vector, get_enhanced_confidence_level

class TestClassificationHelpers(unittest.TestCase):
    def test_prepare_feature_vector_missing_keys(self):
        minimal_results = {
            'ela_mean': 1.0,
            'ela_std': 2.0,
            # Missing many subkeys
        }
        vec = prepare_feature_vector(minimal_results)
        self.assertIsInstance(vec, np.ndarray)
        # Should fill missing features with zeros
        self.assertEqual(vec[0], 1.0)
        self.assertEqual(vec[1], 2.0)
        # Remaining values should be zeros
        self.assertTrue(np.all(vec[2:] == 0))

    def test_get_enhanced_confidence_level(self):
        self.assertEqual(get_enhanced_confidence_level(80), "High")
        self.assertEqual(get_enhanced_confidence_level(65), "Medium")
        self.assertEqual(get_enhanced_confidence_level(40), "Low")


if __name__ == '__main__':
    unittest.main()
