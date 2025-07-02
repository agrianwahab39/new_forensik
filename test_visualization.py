import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from visualization.py
from visualization import (
    visualize_results_advanced, create_core_visuals_grid,
    create_advanced_analysis_grid, create_statistical_grid,
    create_summary_and_validation_grid, create_feature_match_visualization,
    create_block_match_visualization, create_localization_visualization,
    create_frequency_visualization, create_texture_visualization,
    create_edge_visualization, create_illumination_visualization
)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = Image.new('RGB', (200, 200), color='red')
        self.mock_analysis_results = {
            'ela_mean': 15.5,
            'ela_std': 12.3,
            'ela_image': Image.new('L', (200, 200), color=128),
            'ela_regional_stats': {
                'regional_inconsistency': 0.8,
                'outlier_regions': 5,
                'suspicious_regions': []
            },
            'noise_analysis': {
                'overall_inconsistency': 0.6,
                'noise_characteristics': []
            },
            'jpeg_ghost_suspicious_ratio': 0.25,
            'jpeg_ghost': np.random.rand(200, 200),
            'jpeg_analysis': {'quality_responses': [], 'estimated_original_quality': 90},
            'frequency_analysis': {
                'frequency_inconsistency': 1.2,
                'dct_stats': {
                    'low_freq_energy': 1000.0,
                    'mid_freq_energy': 500.0,
                    'high_freq_energy': 250.0,
                    'freq_ratio': 0.5
                }
            },
            'texture_analysis': {
                'overall_inconsistency': 0.4,
                'texture_consistency': {
                    'contrast_consistency': 0.3,
                    'homogeneity_consistency': 0.2,
                    'energy_consistency': 0.4
                }
            },
            'illumination_analysis': {
                'overall_illumination_inconsistency': 0.3
            },
            'edge_analysis': {
                'edge_inconsistency': 0.7
            },
            'statistical_analysis': {
                'overall_entropy': 6.5,
                'R_entropy': 6.2,
                'G_entropy': 6.3,
                'B_entropy': 6.4
            },
            'ransac_inliers': 20,
            'block_matches': [
                {'block1': (10, 10), 'block2': (50, 50)},
                {'block1': (20, 20), 'block2': (60, 60)}
            ],
            'sift_matches': 150,
            'sift_keypoints': None,  # Will be mocked when needed
            'ransac_matches': None,  # Will be mocked when needed
            'localization_analysis': {
                'tampering_percentage': 15.5,
                'combined_tampering_mask': np.random.rand(200, 200) > 0.8
            },
            'classification': {
                'type': 'Copy-Move Forgery',
                'confidence': 'High',
                'copy_move_score': 75,
                'splicing_score': 45,
                'details': ['RANSAC verification found matches', 'Block matching detected']
            },
            'metadata': {
                'Filename': 'test.jpg',
                'FileSize (bytes)': 1024000,
                'DateTime': '2025:01:01 12:00:00',
                'ImageWidth': 200,
                'ImageHeight': 200
            },
            'noise_map': np.random.rand(200, 200)
        }

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        # Close any open matplotlib figures
        plt.close('all')

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('visualization.create_core_visuals_grid')
    @patch('visualization.create_advanced_analysis_grid')
    @patch('visualization.create_statistical_grid')
    @patch('visualization.create_summary_and_validation_grid')
    def test_visualize_results_advanced_success(self, mock_summary_grid, mock_stat_grid,
                                              mock_adv_grid, mock_core_grid,
                                              mock_close, mock_savefig):
        """Test successful visualization creation"""
        output_file = os.path.join(self.temp_dir, "test_visualization.png")
        
        result = visualize_results_advanced(self.test_image, self.mock_analysis_results, output_file)
        
        # Verify that the function returned the filename
        self.assertEqual(result, output_file)
        
        # Verify that all grid creation functions were called
        mock_core_grid.assert_called_once()
        mock_adv_grid.assert_called_once()
        mock_stat_grid.assert_called_once()
        mock_summary_grid.assert_called_once()
        
        # Verify that savefig and close were called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('builtins.print')
    def test_visualize_results_advanced_error(self, mock_print, mock_close, mock_savefig):
        """Test visualization creation with error"""
        mock_savefig.side_effect = Exception("Save error")
        
        result = visualize_results_advanced(self.test_image, self.mock_analysis_results)
        
        # Should return None on error
        self.assertIsNone(result)
        
        # Should still close the figure
        mock_close.assert_called_once()

    def test_create_core_visuals_grid(self):
        """Test core visuals grid creation"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        try:
            create_core_visuals_grid(ax1, ax2, ax3, ax4, self.test_image, self.mock_analysis_results)
            
            # Check that titles were set
            self.assertTrue(ax1.get_title())
            self.assertTrue(ax2.get_title())
            self.assertTrue(ax3.get_title())
            self.assertTrue(ax4.get_title())
            
        finally:
            plt.close(fig)

    def test_create_advanced_analysis_grid(self):
        """Test advanced analysis grid creation"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        try:
            create_advanced_analysis_grid(ax1, ax2, ax3, ax4, self.test_image, self.mock_analysis_results)
            
            # Check that titles were set
            self.assertTrue(ax1.get_title())
            self.assertTrue(ax2.get_title())
            self.assertTrue(ax3.get_title())
            self.assertTrue(ax4.get_title())
            
        finally:
            plt.close(fig)

    def test_create_statistical_grid(self):
        """Test statistical grid creation"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        try:
            create_statistical_grid(ax1, ax2, ax3, ax4, self.mock_analysis_results)
            
            # Check that titles were set
            self.assertTrue(ax1.get_title())
            self.assertTrue(ax2.get_title())
            self.assertTrue(ax3.get_title())
            self.assertTrue(ax4.get_title())
            
        finally:
            plt.close(fig)

    @patch('visualization.create_summary_report')
    @patch('visualization.populate_validation_visuals')
    def test_create_summary_and_validation_grid(self, mock_validation, mock_summary):
        """Test summary and validation grid creation"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        create_summary_and_validation_grid(ax1, ax2, ax3, self.mock_analysis_results)
        
        # Verify that the helper functions were called
        mock_summary.assert_called_once_with(ax1, self.mock_analysis_results)
        mock_validation.assert_called_once_with(ax2, ax3)
        
        plt.close(fig)

    def test_create_feature_match_visualization(self):
        """Test feature match visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_feature_match_visualization(ax, self.test_image, self.mock_analysis_results)
            
            # Check that title was set
            self.assertTrue(ax.get_title())
            self.assertIn("Feature Matches", ax.get_title())
            
        finally:
            plt.close(fig)

    def test_create_block_match_visualization(self):
        """Test block match visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_block_match_visualization(ax, self.test_image, self.mock_analysis_results)
            
            # Check that title was set
            self.assertTrue(ax.get_title())
            self.assertIn("Block Matches", ax.get_title())
            
        finally:
            plt.close(fig)

    def test_create_localization_visualization(self):
        """Test localization visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_localization_visualization(ax, self.test_image, self.mock_analysis_results)
            
            # Check that title was set
            self.assertTrue(ax.get_title())
            self.assertIn("Localization", ax.get_title())
            
        finally:
            plt.close(fig)

    def test_create_frequency_visualization(self):
        """Test frequency visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_frequency_visualization(ax, self.mock_analysis_results)
            
            # Check that title was set
            self.assertTrue(ax.get_title())
            self.assertIn("Frekuensi", ax.get_title())
            
            # Check that y-label was set
            self.assertTrue(ax.get_ylabel())
            
        finally:
            plt.close(fig)

    def test_create_texture_visualization(self):
        """Test texture visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_texture_visualization(ax, self.mock_analysis_results)
            
            # Check that title was set
            self.assertTrue(ax.get_title())
            self.assertIn("Tekstur", ax.get_title())
            
            # Check that x-label was set
            self.assertTrue(ax.get_xlabel())
            
        finally:
            plt.close(fig)

    def test_create_edge_visualization(self):
        """Test edge visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_edge_visualization(ax, self.test_image, self.mock_analysis_results)
            
            # Check that title was set
            self.assertTrue(ax.get_title())
            self.assertIn("Tepi", ax.get_title())
            
        finally:
            plt.close(fig)

    def test_create_illumination_visualization(self):
        """Test illumination visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_illumination_visualization(ax, self.test_image, self.mock_analysis_results)
            
            # Check that title was set
            self.assertTrue(ax.get_title())
            self.assertIn("Iluminasi", ax.get_title())
            
        finally:
            plt.close(fig)

    def test_visualization_with_missing_data(self):
        """Test visualization functions with missing data"""
        incomplete_results = {
            'metadata': {'Filename': 'test.jpg'},
            'ela_image': np.random.rand(100, 100),
            'ela_mean': 10.0
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            # These should not crash even with missing data
            create_frequency_visualization(ax, incomplete_results)
            self.assertTrue(ax.get_title())
            
        finally:
            plt.close(fig)

    def test_visualization_with_none_values(self):
        """Test visualization functions with None values in results"""
        results_with_none = self.mock_analysis_results.copy()
        results_with_none['sift_keypoints'] = None
        results_with_none['ransac_matches'] = None
        results_with_none['block_matches'] = None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            create_feature_match_visualization(ax, self.test_image, results_with_none)
            create_block_match_visualization(ax, self.test_image, results_with_none)
            
            # Should not crash and should have titles
            self.assertTrue(ax.get_title())
            
        finally:
            plt.close(fig)

    def test_mock_analysis_results_structure(self):
        """Test that our mock analysis results have the expected structure"""
        # Test required keys for visualization functions
        required_keys = [
            'ela_mean', 'ela_image', 'metadata', 'frequency_analysis',
            'texture_analysis', 'illumination_analysis', 'edge_analysis'
        ]
        
        for key in required_keys:
            self.assertIn(key, self.mock_analysis_results)

    def test_numpy_array_shapes(self):
        """Test that numpy arrays in mock results have proper shapes"""
        # ela_image is now a PIL Image, so check that it's an Image object
        self.assertIsInstance(self.mock_analysis_results['ela_image'], Image.Image)
        self.assertEqual(self.mock_analysis_results['jpeg_ghost'].ndim, 2)
        self.assertEqual(self.mock_analysis_results['noise_map'].ndim, 2)

    def test_export_kmeans_visualization(self):
        """Test export_kmeans_visualization success and failure"""
        from visualization import export_kmeans_visualization
        # Failure when data missing
        result_none = export_kmeans_visualization(self.test_image, {}, "test.jpg")
        self.assertIsNone(result_none)

        # Success with minimal data
        data = {
            'localization_analysis': {
                'kmeans_localization': {
                    'localization_map': np.zeros((10,10)),
                    'tampering_mask': np.zeros((10,10))
                },
                'combined_tampering_mask': np.zeros((10,10))
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "km.png")
            result = export_kmeans_visualization(self.test_image, data, out_file)
            self.assertEqual(result, out_file)
            self.assertTrue(os.path.exists(out_file))

if __name__ == '__main__':
    unittest.main()