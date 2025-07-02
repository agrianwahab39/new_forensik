import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import numpy as np
from PIL import Image

# Add current directory to sys.path
sys.path.append(os.path.abspath('.'))

# Import main analysis function
from main import analyze_image_comprehensive_advanced

class TestMainAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_image_path = 'im11f.jpg'
        # Create a mock image file
        self.mock_image = Image.new('RGB', (100, 100), color='red')
        self.mock_image.save(self.test_image_path)

    def tearDown(self):
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    @patch('main.save_analysis_to_history')
    @patch('main.validate_image_file')
    @patch('main.extract_enhanced_metadata')
    @patch('main.advanced_preprocess_image')
    @patch('main.perform_multi_quality_ela')
    @patch('main.extract_multi_detector_features')
    @patch('main.detect_copy_move_advanced')
    @patch('main.advanced_tampering_localization')
    @patch('main.classify_manipulation_advanced')
    def test_analyze_image_comprehensive_advanced(self,
                                                  mock_classify,
                                                  mock_localization,
                                                  mock_detect_copy_move,
                                                  mock_extract_features,
                                                  mock_perform_ela,
                                                  mock_preprocess_image,
                                                  mock_extract_metadata,
                                                  mock_validate,
                                                  mock_save_history):
        """Test main analysis pipeline"""
        # Set up mocks
        mock_validate.return_value = None
        mock_extract_metadata.return_value = {'Metadata_Authenticity_Score': 100, 'Filename': self.test_image_path}
        mock_preprocess_image.return_value = (self.mock_image, self.mock_image)
        mock_perform_ela.return_value = (self.mock_image, 10.0, 5.0, {
            'outlier_regions': 0, 'regional_inconsistency': 0, 'suspicious_regions': []}, [], np.zeros((100, 100)))
        mock_extract_features.return_value = ({'sift': ([], None), 'orb': ([], None), 'akaze': ([], None)},
                                               np.ones((100, 100), dtype=np.uint8),
                                               np.array(self.mock_image.convert('L')))
        mock_detect_copy_move.return_value = ([], 0, None)
        mock_localization.return_value = {
            'combined_tampering_mask': np.zeros((100, 100), dtype=bool),
            'tampering_percentage': 0.0,
            'kmeans_localization': {'cluster_ela_means': []} # Ensure sub-key exists
        }
        mock_classify.return_value = {
            'type': 'Authentic', 
            'confidence': 'High',
            'copy_move_score': 25,
            'splicing_score': 30,
            'details': ['Analysis completed successfully'],
            'feature_vector': [0]*28,
            'ml_scores': {'ensemble_copy_move':0, 'ensemble_splicing':0, 'detailed_ml_scores':{}},
            'traditional_scores': {'copy_move': 0, 'splicing': 0}
        }
        mock_save_history.return_value = None

        # To keep the test fast and focused, mock the other internal analysis stages
        with patch('main.detect_copy_move_blocks', return_value=[]), \
             patch('main.analyze_noise_consistency', return_value={'overall_inconsistency': 0.0, 'outlier_count': 0}), \
             patch('main.advanced_jpeg_analysis', return_value={'response_variance':0.0, 'double_compression_indicator': 0.0, 'estimated_original_quality':0, 'compression_inconsistency':False}), \
             patch('main.jpeg_ghost_analysis', return_value=(np.zeros((100, 100)), np.zeros((100, 100), dtype=bool), {})), \
             patch('main.analyze_frequency_domain', return_value={'frequency_inconsistency':0.0, 'dct_stats': {'freq_ratio':0.0}}), \
             patch('main.analyze_texture_consistency', return_value={'overall_inconsistency':0.0, 'texture_features':[], 'texture_consistency':{}}), \
             patch('main.analyze_edge_consistency', return_value={'edge_inconsistency':0.0, 'edge_densities':[]}), \
             patch('main.analyze_illumination_consistency', return_value={'overall_illumination_inconsistency':0.0}), \
             patch('main.perform_statistical_analysis', return_value={'overall_entropy':0.0, 'R_entropy': 0.0, 'G_entropy': 0.0, 'B_entropy': 0.0, 'rg_correlation': 0.0, 'rb_correlation': 0.0, 'gb_correlation': 0.0}):

            # Run the analysis
            output_dir = './results'
            result = analyze_image_comprehensive_advanced(self.test_image_path, output_dir)

            # Check that no exceptions were raised and result is properly returned
            self.assertIsNotNone(result)

            # Check that all main mocks were called
            mock_validate.assert_called_once()
            mock_extract_metadata.assert_called_once()
            mock_preprocess_image.assert_called_once()
            mock_perform_ela.assert_called_once()
            mock_extract_features.assert_called_once()
            mock_detect_copy_move.assert_called_once()
            mock_localization.assert_called_once()
            mock_classify.assert_called_once()
            mock_save_history.assert_called_once()


if __name__ == '__main__':
    unittest.main()