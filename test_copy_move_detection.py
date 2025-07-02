import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
from PIL import Image

# Add current directory to sys.path
sys.path.append(os.path.abspath('.'))

# Import functions from copy_move_detection.py
from copy_move_detection import (
    detect_copy_move_sift, detect_copy_move_orb, detect_copy_move_advanced,
    detect_copy_move_blocks
)

class TestCopyMoveDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = Image.new('RGB', (200, 200), color='gray')

    def test_detect_copy_move_sift(self):
        """Test SIFT-based copy-move detector"""
        keypoints, descriptors, matches = detect_copy_move_sift(self.test_image)
        
        # Check that keypoints and matches are lists
        self.assertIsInstance(keypoints, list)
        self.assertIsInstance(matches, list)

    def test_detect_copy_move_orb(self):
        """Test ORB-based copy-move detector"""
        keypoints, descriptors, matches = detect_copy_move_orb(self.test_image)
        
        # Check that keypoints and matches are lists
        self.assertIsInstance(keypoints, list)
        self.assertIsInstance(matches, list)

    @patch('copy_move_detection.match_sift_features')
    def test_detect_copy_move_advanced_feature_sets(self, mock_match_features):
        """Test advanced copy-move detection with feature sets"""
        mock_match_features.return_value = ([], 0, None)
        
        feature_sets = {
            'sift': ([], None),
            'orb': ([], None),
            'akaze': ([], None)
        }
        
        matches, inliers, transform = detect_copy_move_advanced(feature_sets, self.test_image.size)
        
        # Test that result is a tuple
        self.assertIsInstance(matches, list)
        self.assertEqual(inliers, 0)
        self.assertIsNone(transform)

    def test_detect_copy_move_blocks(self):
        """Test block-based copy-move detector"""
        matches = detect_copy_move_blocks(self.test_image)
        
        # Check that matches is a list
        self.assertIsInstance(matches, list)

if __name__ == '__main__':
    unittest.main()