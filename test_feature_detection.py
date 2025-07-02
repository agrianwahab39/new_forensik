import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
from PIL import Image
import cv2

# Add current directory to sys.path
sys.path.append(os.path.abspath('.'))

# Import functions from feature_detection.py
from feature_detection import (
    extract_multi_detector_features, match_sift_features,
    match_orb_features, match_akaze_features
)

class TestFeatureDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = Image.new('RGB', (200, 200), color='gray')
        self.test_ela_image = Image.new('L', (200, 200), color=128)
        self.ela_mean = 50.0
        self.ela_stddev = 15.0
        
        # Mock keypoints and descriptors
        self.mock_keypoints = [cv2.KeyPoint(x=50, y=50, size=10), cv2.KeyPoint(x=100, y=100, size=10)]
        self.mock_descriptors = np.random.rand(2, 128).astype(np.float32)

    @patch('cv2.SIFT_create')
    @patch('cv2.ORB_create')
    @patch('cv2.AKAZE_create')
    def test_extract_multi_detector_features(self, mock_akaze, mock_orb, mock_sift):
        """Test multi-detector feature extraction"""
        # Mock detector instances
        mock_sift_instance = MagicMock()
        mock_orb_instance = MagicMock()
        mock_akaze_instance = MagicMock()
        
        mock_sift.return_value = mock_sift_instance
        mock_orb.return_value = mock_orb_instance
        mock_akaze.return_value = mock_akaze_instance
        
        # Mock detectAndCompute returns
        mock_sift_instance.detectAndCompute.return_value = (self.mock_keypoints, self.mock_descriptors)
        mock_orb_instance.detectAndCompute.return_value = (self.mock_keypoints, self.mock_descriptors)
        mock_akaze_instance.detectAndCompute.return_value = (self.mock_keypoints, self.mock_descriptors)
        
        feature_sets, roi_mask, gray_enhanced = extract_multi_detector_features(
            self.test_image, self.test_ela_image, self.ela_mean, self.ela_stddev)
        
        # Check that feature sets are returned
        self.assertIn('sift', feature_sets)
        self.assertIn('orb', feature_sets)
        self.assertIn('akaze', feature_sets)
        
        # Check that ROI mask and enhanced gray image are returned
        self.assertIsNotNone(roi_mask)
        self.assertIsNotNone(gray_enhanced)
        
        # Check that detectors were called
        mock_sift_instance.detectAndCompute.assert_called_once()
        mock_orb_instance.detectAndCompute.assert_called_once()
        mock_akaze_instance.detectAndCompute.assert_called_once()

    @patch('cv2.AKAZE_create')
    def test_extract_features_akaze_exception(self, mock_akaze):
        """Test feature extraction when AKAZE throws exception"""
        mock_akaze.side_effect = Exception("AKAZE error")
        
        with patch('cv2.SIFT_create') as mock_sift:
            with patch('cv2.ORB_create') as mock_orb:
                mock_sift_instance = MagicMock()
                mock_orb_instance = MagicMock()
                mock_sift.return_value = mock_sift_instance
                mock_orb.return_value = mock_orb_instance
                
                mock_sift_instance.detectAndCompute.return_value = (self.mock_keypoints, self.mock_descriptors)
                mock_orb_instance.detectAndCompute.return_value = (self.mock_keypoints, self.mock_descriptors)
                
                feature_sets, roi_mask, gray_enhanced = extract_multi_detector_features(
                    self.test_image, self.test_ela_image, self.ela_mean, self.ela_stddev)
                
                # AKAZE should return empty results on exception
                self.assertEqual(feature_sets['akaze'], ([], None))

    @patch('cv2.FlannBasedMatcher')
    @patch('cv2.estimateAffine2D')
    def test_match_sift_features(self, mock_estimate_affine, mock_flann):
        """Test SIFT feature matching"""
        # Mock FLANN matcher
        mock_matcher = MagicMock()
        mock_flann.return_value = mock_matcher
        
        # Mock matches
        mock_match1 = MagicMock()
        mock_match1.trainIdx = 1
        mock_match1.distance = 0.5
        
        mock_matcher.knnMatch.return_value = [
            [mock_match1, mock_match1]  # Self-match will be skipped
        ]
        
        # Mock RANSAC
        mock_estimate_affine.return_value = (np.eye(2, 3), np.array([[1], [1]]))
        
        matches, inliers, transform = match_sift_features(
            self.mock_keypoints, self.mock_descriptors, 0.75, 40, 5.0, 3)
        
        # Check that matches are returned
        self.assertIsInstance(matches, list)
        self.assertIsInstance(inliers, int)

    def test_match_sift_features_insufficient_matches(self):
        """Test SIFT matching with insufficient matches"""
        # Empty descriptors should return empty results
        empty_descriptors = np.array([]).reshape(0, 128).astype(np.float32)
        empty_keypoints = []
        
        matches, inliers, transform = match_sift_features(
            empty_keypoints, empty_descriptors, 0.75, 40, 5.0, 3)
        
        self.assertEqual(matches, [])
        self.assertEqual(inliers, 0)
        self.assertIsNone(transform)

    @patch('cv2.FlannBasedMatcher')
    @patch('cv2.findHomography')
    def test_match_sift_features_homography(self, mock_homography, mock_flann):
        """Test SIFT matching path when homography succeeds"""
        mock_matcher = MagicMock()
        mock_flann.return_value = mock_matcher
        m = MagicMock()
        m.trainIdx = 1
        m.distance = 0.5
        mock_matcher.knnMatch.return_value = [[m, m]]
        mock_homography.return_value = (np.eye(3), np.array([[1],[1]]))
        matches, inliers, transform = match_sift_features(
            self.mock_keypoints, self.mock_descriptors, 0.75, 40, 5.0, 1)
        self.assertEqual(inliers, 2)
        self.assertEqual(transform[0], 'homography')

    @patch('cv2.FlannBasedMatcher')
    @patch('cv2.estimateAffine2D')
    @patch('cv2.findHomography')
    @patch('cv2.estimateAffinePartial2D')
    def test_match_sift_features_no_inliers(self, mock_partial, mock_homography,
                                            mock_affine, mock_flann):
        """Return no transform when all estimations fail"""
        mock_matcher = MagicMock()
        mock_flann.return_value = mock_matcher
        m = MagicMock()
        m.trainIdx = 1
        m.distance = 0.5
        mock_matcher.knnMatch.return_value = [[m, m]]
        mock_affine.return_value = (None, None)
        mock_homography.return_value = (None, None)
        mock_partial.return_value = (None, None)
        matches, inliers, transform = match_sift_features(
            self.mock_keypoints, self.mock_descriptors, 0.75, 40, 5.0, 1)
        self.assertIsNone(transform)

    @patch('cv2.BFMatcher')
    def test_match_orb_features(self, mock_bf_matcher):
        """Test ORB feature matching"""
        mock_matcher = MagicMock()
        mock_bf_matcher.return_value = mock_matcher

        # We need to provide enough keypoints so distances can be calculated
        keypoints = [cv2.KeyPoint(10, 10, 1), cv2.KeyPoint(110, 110, 1), cv2.KeyPoint(120, 20, 1), cv2.KeyPoint(30, 130, 1)]
        descriptors = np.random.rand(4, 32).astype(np.uint8)

        # Mock matches for keypoint 0
        # These matches will pass the distance check (> 40)
        m1 = MagicMock(trainIdx=1, distance=50) # queryIdx=0, trainIdx=1
        m2 = MagicMock(trainIdx=2, distance=60) # queryIdx=0, trainIdx=2
        m3 = MagicMock(trainIdx=3, distance=70) # queryIdx=0, trainIdx=3

        # Return matches for keypoint 0 that are not self-matches
        # This will result in 3 match pairs, satisfying min_inliers=3
        mock_matcher.knnMatch.return_value = [
             [MagicMock(trainIdx=0), m1, m2, m3] # Matches for kp[0], first is self-match
        ] * len(keypoints) # simulate matches for all keypoints
        
        matches, inliers, transform = match_orb_features(
            keypoints, descriptors, min_distance=40, ransac_thresh=5.0, min_inliers=3)
        
        # Check that matches are returned
        self.assertIsInstance(matches, list)
        self.assertGreaterEqual(inliers, 3, "Should have enough inliers")
        self.assertIsInstance(transform, tuple, "Transform should be a tuple")

    def test_match_orb_features_insufficient_matches(self):
        """Test ORB matching with insufficient matches"""
        with patch('cv2.BFMatcher') as mock_bf_matcher:
            mock_matcher = MagicMock()
            mock_bf_matcher.return_value = mock_matcher
            mock_matcher.knnMatch.return_value = []  # No matches
            
            matches, inliers, transform = match_orb_features(
                self.mock_keypoints, self.mock_descriptors, 40, 5.0, 10)  # High min_inliers
            
            self.assertEqual(matches, [])
            self.assertEqual(inliers, 0)
            self.assertIsNone(transform)

    @patch('cv2.BFMatcher')
    def test_match_akaze_features(self, mock_bf_matcher):
        """Test AKAZE feature matching"""
        mock_matcher = MagicMock()
        mock_bf_matcher.return_value = mock_matcher
        
        # Mock matches
        mock_match = MagicMock()
        mock_match.trainIdx = 1
        mock_match.distance = 50
        
        mock_matcher.knnMatch.return_value = [
            [mock_match, mock_match]  # Self-match will be skipped
        ]
        
        matches, inliers, transform = match_akaze_features(
            self.mock_keypoints, self.mock_descriptors, 40, 5.0, 3)
        
        # Check that matches are returned
        self.assertIsInstance(matches, list)
        self.assertIsInstance(inliers, int)
        self.assertIsInstance(transform, tuple)

    def test_match_akaze_features_none_descriptors(self):
        """Test AKAZE matching with None descriptors"""
        matches, inliers, transform = match_akaze_features(
            self.mock_keypoints, None, 40, 5.0, 3)
        
        self.assertEqual(matches, [])
        self.assertEqual(inliers, 0)
        self.assertIsNone(transform)

    def test_roi_mask_generation(self):
        """Test that ROI mask is generated correctly"""
        with patch('cv2.SIFT_create') as mock_sift:
            with patch('cv2.ORB_create') as mock_orb:
                with patch('cv2.AKAZE_create') as mock_akaze:
                    # Mock all detectors
                    for mock_detector in [mock_sift, mock_orb, mock_akaze]:
                        mock_instance = MagicMock()
                        mock_detector.return_value = mock_instance
                        mock_instance.detectAndCompute.return_value = ([], None)
                    
                    # Test with high ELA values to create ROI
                    high_ela_image = Image.new('L', (200, 200), color=200)
                    
                    feature_sets, roi_mask, gray_enhanced = extract_multi_detector_features(
                        self.test_image, high_ela_image, 50.0, 20.0)
                    
                    # ROI mask should have some non-zero values
                    self.assertEqual(roi_mask.shape, (200, 200))
                    self.assertTrue(np.any(roi_mask > 0))

if __name__ == '__main__':
    unittest.main()