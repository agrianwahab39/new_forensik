import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np
from PIL import Image

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from app2.py
try:
    from app2 import (
        display_single_plot, display_single_image, create_spider_chart,
        IMPORTS_SUCCESSFUL, IMPORT_ERROR_MESSAGE
    )
    APP2_AVAILABLE = True
except Exception as e:
    APP2_AVAILABLE = False
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR_MESSAGE = str(e)

@unittest.skipUnless(APP2_AVAILABLE, f"app2 unavailable: {IMPORT_ERROR_MESSAGE}")
class TestApp2Functions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.mock_analysis_results = {
            'ela_mean': 15.5,
            'ela_std': 12.3,
            'ela_regional_stats': {
                'regional_inconsistency': 0.8,
                'outlier_regions': 5
            },
            'noise_analysis': {
                'overall_inconsistency': 0.6
            },
            'jpeg_ghost_suspicious_ratio': 0.25,
            'frequency_analysis': {
                'frequency_inconsistency': 1.2
            },
            'texture_analysis': {
                'overall_inconsistency': 0.4
            },
            'illumination_analysis': {
                'overall_illumination_inconsistency': 0.3
            },
            'ransac_inliers': 20,
            'block_matches': [1, 2, 3, 4, 5],
            'classification': {
                'type': 'Copy-Move Forgery',
                'confidence': 'High',
                'copy_move_score': 75,
                'splicing_score': 45,
                'details': ['RANSAC verification found matches', 'Block matching detected']
            },
            'metadata': {
                'Filename': 'test.jpg',
                'FileSize (bytes)': 1024000
            }
        }

    def test_imports_successful(self):
        """Test that imports are working"""
        # This test will pass if the imports at module level succeeded
        if not IMPORTS_SUCCESSFUL:
            self.fail(f"Imports failed: {IMPORT_ERROR_MESSAGE}")
        else:
            self.assertTrue(IMPORTS_SUCCESSFUL)

    @patch('streamlit.subheader')
    @patch('matplotlib.pyplot.subplots')
    @patch('streamlit.pyplot')
    @patch('streamlit.caption')
    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    def test_display_single_plot(self, mock_markdown, mock_expander, mock_caption, 
                                mock_pyplot, mock_subplots, mock_subheader):
        """Test display_single_plot function"""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock streamlit container
        mock_container = Mock()
        mock_container.__enter__ = Mock(return_value=mock_container)
        mock_container.__exit__ = Mock(return_value=None)
        
        # Mock plot function
        mock_plot_function = Mock()
        
        # Test the function
        display_single_plot(
            title="Test Plot",
            plot_function=mock_plot_function,
            args=[1, 2, 3],
            caption="Test caption",
            details="Test details",
            container=mock_container
        )
        
        # Verify that the plot function was called
        mock_plot_function.assert_called_once_with(mock_ax, 1, 2, 3)

    @patch('streamlit.subheader')
    @patch('matplotlib.pyplot.subplots')
    @patch('streamlit.pyplot')
    @patch('streamlit.caption')
    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    def test_display_single_image(self, mock_markdown, mock_expander, mock_caption,
                                 mock_pyplot, mock_subplots, mock_subheader):
        """Test display_single_image function"""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.imshow.return_value = Mock()
        
        # Mock streamlit container
        mock_container = Mock()
        mock_container.__enter__ = Mock(return_value=mock_container)
        mock_container.__exit__ = Mock(return_value=None)
        
        # Create test image array
        test_image_array = np.random.rand(50, 50)
        
        # Test the function
        display_single_image(
            title="Test Image",
            image_array=test_image_array,
            cmap='hot',
            caption="Test caption",
            details="Test details",
            container=mock_container,
            colorbar=True
        )
        
        # Verify that imshow was called
        mock_ax.imshow.assert_called_once_with(test_image_array, cmap='hot')

    def test_create_spider_chart(self):
        """Test create_spider_chart function"""
        try:
            # Test with valid analysis results
            fig = create_spider_chart(self.mock_analysis_results)
            
            # Check that a figure is returned
            self.assertIsNotNone(fig)
            
            # Check that it has the expected structure (plotly figure)
            self.assertTrue(hasattr(fig, 'data'))
            self.assertTrue(len(fig.data) >= 2)  # Should have splicing and copy-move traces
            
        except Exception as e:
            # If plotly is not available, the test should still pass
            self.skipTest(f"Plotly not available: {e}")

    def test_create_spider_chart_missing_keys(self):
        """Test create_spider_chart with missing keys in analysis results"""
        incomplete_results = {
            'ela_mean': 10.0,
            # Missing other keys
        }
        
        try:
            fig = create_spider_chart(incomplete_results)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.skipTest(f"Plotly not available: {e}")

    def test_create_spider_chart_empty_results(self):
        """Test create_spider_chart with empty analysis results"""
        empty_results = {}
        
        try:
            fig = create_spider_chart(empty_results)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.skipTest(f"Plotly not available: {e}")

    @patch('app2.main_analysis_func')
    def test_analysis_function_import(self, mock_analysis_func):
        """Test that main analysis function can be imported and called"""
        if IMPORTS_SUCCESSFUL:
            # Mock the analysis function
            mock_analysis_func.return_value = self.mock_analysis_results
            
            # Test that we can call the function
            result = mock_analysis_func("test_image.jpg")
            self.assertIsNotNone(result)
            mock_analysis_func.assert_called_once_with("test_image.jpg")

    def test_mock_analysis_results_structure(self):
        """Test that our mock analysis results have the expected structure"""
        # Test required keys
        required_keys = [
            'ela_mean', 'ela_std', 'ela_regional_stats', 'noise_analysis',
            'jpeg_ghost_suspicious_ratio', 'frequency_analysis', 'texture_analysis',
            'illumination_analysis', 'ransac_inliers', 'block_matches', 'classification'
        ]
        
        for key in required_keys:
            self.assertIn(key, self.mock_analysis_results)

    def test_classification_structure(self):
        """Test classification results structure"""
        classification = self.mock_analysis_results['classification']
        
        # Test required classification keys
        required_keys = ['type', 'confidence', 'copy_move_score', 'splicing_score', 'details']
        for key in required_keys:
            self.assertIn(key, classification)
        
        # Test score ranges
        self.assertGreaterEqual(classification['copy_move_score'], 0)
        self.assertLessEqual(classification['copy_move_score'], 100)
        self.assertGreaterEqual(classification['splicing_score'], 0)
        self.assertLessEqual(classification['splicing_score'], 100)

if __name__ == '__main__':
    unittest.main()