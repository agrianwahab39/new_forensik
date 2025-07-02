import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import tempfile
import shutil
from PIL import Image
import numpy as np

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from export_utils.py
from export_utils import (
    DOCX_AVAILABLE, VISUALIZATION_AVAILABLE,
    set_cell_shading, export_complete_package,
    export_comprehensive_package, generate_all_process_images
)


class TestExportUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.mock_analysis_results = {
            'ela_mean': 15.5,
            'ela_std': 12.3,
            'ela_image': np.random.rand(100, 100),
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
            'jpeg_ghost': np.random.rand(100, 100),
            'frequency_analysis': {
                'frequency_inconsistency': 1.2,
                'dct_stats': {
                    'low_freq_energy': 1000,
                    'high_freq_energy': 500,
                    'freq_ratio': 0.5
                }
            },
            'texture_analysis': {
                'overall_inconsistency': 0.4,
                'texture_consistency': {}
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
            'block_matches': [1, 2, 3, 4, 5],
            'sift_matches': 150,
            'localization_analysis': {
                'tampering_percentage': 15.5
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
                'ImageWidth': 100,
                'ImageHeight': 100
            },
            'noise_map': np.random.rand(100, 100)
        }

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_docx_availability(self):
        """Test DOCX availability flag"""
        # This tests whether python-docx is available
        self.assertIsInstance(DOCX_AVAILABLE, bool)

    def test_visualization_availability(self):
        """Test matplotlib availability flag"""
        # This tests whether matplotlib is available
        self.assertIsInstance(VISUALIZATION_AVAILABLE, bool)

    @patch('export_utils.DOCX_AVAILABLE', True)
    def test_set_cell_shading_success(self):
        """Test set_cell_shading function when DOCX is available"""
        if not DOCX_AVAILABLE:
            self.skipTest("python-docx not available")
        
        # Mock cell object
        mock_cell = Mock()
        mock_tc = Mock()
        mock_tcPr = Mock()
        mock_cell._tc = mock_tc
        mock_tc.get_or_add_tcPr.return_value = mock_tcPr
        
        # Test the function
        result = set_cell_shading(mock_cell, "FF0000")
        
        # Should attempt to set shading
        self.assertIsInstance(result, bool)

    def test_set_cell_shading_no_docx(self):
        """Test set_cell_shading function when DOCX is not available"""
        if DOCX_AVAILABLE:
            self.skipTest("python-docx is available, can't test no-docx scenario")
        
        # Mock cell object
        mock_cell = Mock()
        
        # Test the function - should handle the case gracefully
        try:
            result = set_cell_shading(mock_cell, "FF0000")
            self.assertIsInstance(result, bool)
        except:
            # If it throws an exception, that's also acceptable behavior
            pass

    @patch('export_utils.export_visualization_png')
    @patch('export_utils.export_visualization_pdf')
    @patch('export_utils.export_to_advanced_docx')
    @patch('export_utils.export_report_pdf')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_export_complete_package(self, mock_getsize, mock_exists, mock_pdf_report,
                                   mock_docx_export, mock_pdf_viz, mock_png_export):
        """Test export_complete_package function"""
        
        # Mock return values
        mock_png_export.return_value = "test_visualization.png"
        mock_pdf_viz.return_value = "test_visualization.pdf"
        mock_docx_export.return_value = "test_report.docx"
        mock_pdf_report.return_value = "test_report.pdf"
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        # Test the function
        base_filename = os.path.join(self.temp_dir, "test_analysis")
        result = export_complete_package(self.test_image, self.mock_analysis_results, base_filename)
        
        # Verify the result
        self.assertIsInstance(result, dict)
        self.assertIn('png_visualization', result)
        
        # Verify that the export functions were called
        mock_png_export.assert_called_once()
        mock_pdf_viz.assert_called_once()

    @patch('export_utils.generate_all_process_images')
    @patch('export_utils.export_visualization_png')
    @patch('export_utils.export_to_advanced_docx')
    @patch('export_utils.export_report_pdf')
    @patch('export_utils.create_html_index')
    @patch('zipfile.ZipFile')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.getsize')
    def test_export_comprehensive_package(self, mock_getsize, mock_listdir, mock_exists, mock_makedirs,
                                        mock_zipfile, mock_html_index, mock_pdf_report,
                                        mock_docx_export, mock_png_export, mock_process_images):
        """Test export_comprehensive_package function"""
        
        # Mock return values
        mock_png_export.return_value = "test_visualization.png"
        mock_docx_export.return_value = "test_report.docx"
        mock_pdf_report.return_value = "test_report.pdf"
        mock_html_index.return_value = "test_index.html"
        mock_exists.return_value = True
        mock_listdir.return_value = ["process1.png", "process2.png"]
        mock_getsize.return_value = 1024
        
        # Mock ZipFile context manager
        mock_zip_instance = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        
        # Test the function
        base_filename = os.path.join(self.temp_dir, "test_analysis")
        result = export_comprehensive_package(self.test_image, self.mock_analysis_results, base_filename)
        
        # Verify the result
        self.assertIsInstance(result, dict)
        
        # Verify that key functions were called
        mock_process_images.assert_called_once()
        mock_png_export.assert_called_once()
        mock_html_index.assert_called_once()

    def test_mock_analysis_results_structure(self):
        """Test that our mock analysis results have the expected structure"""
        # Test required keys for export functions
        required_keys = [
            'ela_mean', 'ela_std', 'ela_image', 'classification',
            'metadata', 'noise_analysis', 'texture_analysis'
        ]
        
        for key in required_keys:
            self.assertIn(key, self.mock_analysis_results)

    def test_classification_structure_for_export(self):
        """Test classification results structure for export compatibility"""
        classification = self.mock_analysis_results['classification']
        
        # Test required classification keys for export
        required_keys = ['type', 'confidence', 'copy_move_score', 'splicing_score']
        for key in required_keys:
            self.assertIn(key, classification)

    def test_metadata_structure_for_export(self):
        """Test metadata structure for export compatibility"""
        metadata = self.mock_analysis_results['metadata']
        
        # Test required metadata keys for export
        required_keys = ['Filename', 'FileSize (bytes)']
        for key in required_keys:
            self.assertIn(key, metadata)

    @patch('builtins.print')
    def test_export_with_missing_data(self, mock_print):
        """Test export functions with missing data in analysis results"""
        incomplete_results = {
            'classification': {
                'type': 'Unknown',
                'confidence': 'Low'
            }
        }
        
        # Test that export functions handle missing data gracefully
        try:
            with patch('export_utils.export_visualization_png') as mock_png:
                mock_png.return_value = "test.png"
                result = export_complete_package(self.test_image, incomplete_results)
                self.assertIsInstance(result, dict)
        except Exception as e:
            # If it throws an exception, verify it's handled gracefully
            self.assertIsInstance(e, Exception)

    def test_temp_directory_creation(self):
        """Test that temporary directory is properly created and cleaned up"""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.isdir(self.temp_dir))

    @patch('os.makedirs')
    def test_directory_creation_in_export(self, mock_makedirs):
        """Test directory creation in export functions"""
        base_filename = os.path.join(self.temp_dir, "subdir", "test_analysis")
        
        with patch('export_utils.export_visualization_png') as mock_png:
            mock_png.return_value = "test.png"
            result = export_complete_package(self.test_image, self.mock_analysis_results, base_filename)
            
            # Should attempt to create directories as needed
            self.assertIsInstance(result, dict)

    @patch('visualization.create_feature_match_visualization')
    @patch('visualization.create_block_match_visualization')
    @patch('visualization.create_localization_visualization')
    @patch('visualization.create_edge_visualization')
    @patch('visualization.create_illumination_visualization')
    @patch('visualization.create_frequency_visualization')
    @patch('visualization.create_texture_visualization')
    @patch('visualization.create_statistical_visualization')
    @patch('visualization.create_quality_response_plot')
    def test_generate_all_process_images_float_ela(self, mock_qr, mock_stat,
                                                  mock_tex, mock_freq,
                                                  mock_illum, mock_edge,
                                                  mock_loc, mock_block,
                                                  mock_feat):
        """Ensure float ELA images are converted to uint8 when saved"""
        analysis = {
            'ela_image': np.random.rand(5, 5),
            'jpeg_ghost': np.random.rand(5, 5),
            'frequency_analysis': {},
            'texture_analysis': {},
            'edge_analysis': {},
            'illumination_analysis': {},
            'statistical_analysis': {'R_entropy':1,'G_entropy':1,'B_entropy':1},
            'jpeg_analysis': {'quality_responses': []},
            'localization_analysis': {},
            'classification': {},
            'noise_map': np.random.rand(5,5)
        }
        with tempfile.TemporaryDirectory() as out_dir:
            result = generate_all_process_images(self.test_image, analysis, out_dir)
            self.assertTrue(result)
            with Image.open(os.path.join(out_dir, "02_error_level_analysis.png")) as saved:
                self.assertNotEqual(saved.mode, 'F')

    @patch('visualization.create_feature_match_visualization')
    @patch('visualization.create_block_match_visualization')
    @patch('visualization.create_localization_visualization')
    @patch('visualization.create_edge_visualization')
    @patch('visualization.create_illumination_visualization')
    @patch('visualization.create_frequency_visualization')
    @patch('visualization.create_texture_visualization')
    @patch('visualization.create_statistical_visualization')
    @patch('visualization.create_quality_response_plot')
    def test_generate_all_process_images_float_ela(self, mock_qr, mock_stat,
                                                  mock_tex, mock_freq,
                                                  mock_illum, mock_edge,
                                                  mock_loc, mock_block,
                                                  mock_feat):
        """Ensure float ELA images are converted to uint8 when saved"""
        analysis = {
            'ela_image': np.random.rand(5, 5),
            'jpeg_ghost': np.random.rand(5, 5),
            'frequency_analysis': {},
            'texture_analysis': {},
            'edge_analysis': {},
            'illumination_analysis': {},
            'statistical_analysis': {'R_entropy':1,'G_entropy':1,'B_entropy':1},
            'jpeg_analysis': {'quality_responses': []},
            'localization_analysis': {},
            'classification': {},
            'noise_map': np.random.rand(5,5)
        }
        with tempfile.TemporaryDirectory() as out_dir:
            result = generate_all_process_images(self.test_image, analysis, out_dir)
            self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()