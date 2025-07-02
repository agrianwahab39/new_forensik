import unittest
from unittest.mock import Mock, patch, mock_open
import sys
import os
import tempfile
import shutil
import json
import numpy as np

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from utils.py
from utils import (
    detect_outliers_iqr, calculate_skewness, calculate_kurtosis,
    normalize_array, safe_divide, load_analysis_history,
    save_analysis_to_history, delete_all_history, delete_selected_history,
    get_history_count, clear_empty_thumbnail_folder,
    HISTORY_FILE, THUMBNAIL_DIR
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])  # Data with one clear outlier
        self.normal_data = np.array([1, 2, 3, 4, 5])
        self.constant_data = np.array([5, 5, 5, 5, 5])
        
        # Mock history data
        self.mock_history = [
            {
                'timestamp': '2025-01-01 12:00:00',
                'image_name': 'test1.jpg',
                'thumbnail_path': 'thumb1.jpg',
                'analysis_summary': {'type': 'Copy-Move'},
                'processing_time': 10.5
            },
            {
                'timestamp': '2025-01-01 13:00:00',
                'image_name': 'test2.jpg',
                'thumbnail_path': 'thumb2.jpg',
                'analysis_summary': {'type': 'Splicing'},
                'processing_time': 12.3
            }
        ]

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection"""
        outliers = detect_outliers_iqr(self.test_data)
        
        # Should detect outliers at higher values
        self.assertIsInstance(outliers, np.ndarray)
        # Should find outliers in data with extreme values (100, 150, 200 vs 1-5)
        self.assertGreaterEqual(len(outliers), 1)  # At least one outlier expected
        
        # Test with normal data (fewer outliers expected)
        no_outliers = detect_outliers_iqr(self.normal_data)
        self.assertLessEqual(len(no_outliers), 1)  # Allow for edge cases
        
        # Test with custom factor
        outliers_custom = detect_outliers_iqr(self.test_data, factor=3.0)
        self.assertTrue(len(outliers_custom) <= len(outliers))  # Stricter threshold

        # Empty array should return empty array
        self.assertEqual(len(detect_outliers_iqr(np.array([]))), 0)

    def test_calculate_skewness(self):
        """Test skewness calculation"""
        # Test with normal data
        skewness = calculate_skewness(self.normal_data)
        self.assertIsInstance(skewness, (int, float, np.floating))
        
        # Test with constant data (should return 0)
        skewness_constant = calculate_skewness(self.constant_data)
        self.assertEqual(skewness_constant, 0)
        
        # Test with single value
        single_value = np.array([5])
        skewness_single = calculate_skewness(single_value)
        self.assertEqual(skewness_single, 0)

    def test_calculate_kurtosis(self):
        """Test kurtosis calculation"""
        # Test with normal data
        kurtosis = calculate_kurtosis(self.normal_data)
        self.assertIsInstance(kurtosis, (int, float, np.floating))
        
        # Test with constant data (should return 0)
        kurtosis_constant = calculate_kurtosis(self.constant_data)
        self.assertEqual(kurtosis_constant, 0)
        
        # Test with single value
        single_value = np.array([5])
        kurtosis_single = calculate_kurtosis(single_value)
        self.assertEqual(kurtosis_single, 0)

    def test_skewness_kurtosis_empty(self):
        """Functions should handle empty arrays gracefully"""
        empty = np.array([])
        self.assertEqual(calculate_skewness(empty), 0)
        self.assertEqual(calculate_kurtosis(empty), 0)

    def test_normalize_array(self):
        """Test array normalization"""
        # Test with normal data
        normalized = normalize_array(self.test_data)
        
        self.assertEqual(normalized.shape, self.test_data.shape)
        self.assertAlmostEqual(np.min(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.max(normalized), 1.0, places=10)
        
        # Test with constant data (should return zeros)
        normalized_constant = normalize_array(self.constant_data)
        self.assertTrue(np.all(normalized_constant == 0))
        
        # Test with 2D array
        test_2d = np.array([[1, 2], [3, 4]])
        normalized_2d = normalize_array(test_2d)
        self.assertEqual(normalized_2d.shape, test_2d.shape)

    def test_safe_divide(self):
        """Test safe division function"""
        # Normal division
        result = safe_divide(10, 2)
        self.assertEqual(result, 5.0)
        
        # Division by zero with default
        result_zero = safe_divide(10, 0)
        self.assertEqual(result_zero, 0.0)
        
        # Division by zero with custom default
        result_custom = safe_divide(10, 0, default=99)
        self.assertEqual(result_custom, 99)
        
        # Float inputs
        result_float = safe_divide(7.5, 2.5)
        self.assertEqual(result_float, 3.0)

    @patch('builtins.open', new_callable=mock_open, read_data='[]')
    @patch('os.path.exists')
    def test_load_analysis_history_empty(self, mock_exists, mock_file):
        """Test loading empty history"""
        mock_exists.return_value = True
        
        history = load_analysis_history()
        self.assertEqual(history, [])

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    @patch('os.path.exists')
    def test_load_analysis_history_invalid_json(self, mock_exists, mock_file):
        """Test loading history with invalid JSON"""
        mock_exists.return_value = True
        
        with patch('builtins.print'):  # Suppress warning prints
            history = load_analysis_history()
            self.assertEqual(history, [])

    @patch('os.path.exists')
    def test_load_analysis_history_no_file(self, mock_exists):
        """Test loading history when file doesn't exist"""
        mock_exists.return_value = False
        
        history = load_analysis_history()
        self.assertEqual(history, [])

    @patch('utils.load_analysis_history')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_analysis_to_history(self, mock_json_dump, mock_file, mock_load):
        """Test saving analysis to history"""
        mock_load.return_value = self.mock_history.copy()
        
        save_analysis_to_history(
            image_name="new_test.jpg",
            analysis_summary={"type": "Authentic"},
            processing_time=8.5,
            thumbnail_path="new_thumb.jpg"
        )
        
        # Verify that json.dump was called
        mock_json_dump.assert_called_once()
        
        # Verify the structure of saved data
        saved_data = mock_json_dump.call_args[0][0]
        self.assertEqual(len(saved_data), 3)  # Original 2 + new 1
        self.assertEqual(saved_data[-1]['image_name'], "new_test.jpg")

    @patch('os.path.exists')
    @patch('os.remove')
    @patch('shutil.rmtree')
    @patch('builtins.print')
    def test_delete_all_history_success(self, mock_print, mock_rmtree, mock_remove, mock_exists):
        """Test successful deletion of all history"""
        mock_exists.return_value = True
        
        result = delete_all_history()
        
        self.assertTrue(result)
        mock_remove.assert_called_once()
        mock_rmtree.assert_called_once()

    @patch('os.path.exists')
    @patch('os.remove')
    @patch('builtins.print')
    def test_delete_all_history_error(self, mock_print, mock_remove, mock_exists):
        """Test error handling in delete_all_history"""
        mock_exists.return_value = True
        mock_remove.side_effect = OSError("Permission denied")
        
        result = delete_all_history()
        
        self.assertFalse(result)

    @patch('utils.load_analysis_history')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('builtins.print')
    def test_delete_selected_history_success(self, mock_print, mock_os_remove, 
                                           mock_exists, mock_json_dump, mock_file, mock_load):
        """Test successful deletion of selected history entries"""
        mock_load.return_value = self.mock_history.copy()
        mock_exists.return_value = True
        
        # Delete index 0
        result = delete_selected_history([0])
        
        self.assertTrue(result)
        mock_json_dump.assert_called_once()
        
        # Verify remaining data
        saved_data = mock_json_dump.call_args[0][0]
        self.assertEqual(len(saved_data), 1)
        self.assertEqual(saved_data[0]['image_name'], 'test2.jpg')

    @patch('utils.load_analysis_history')
    def test_delete_selected_history_empty(self, mock_load):
        """Test deleting from empty history"""
        mock_load.return_value = []
        
        with patch('builtins.print'):
            result = delete_selected_history([0])
            self.assertFalse(result)

    @patch('utils.load_analysis_history')
    def test_delete_selected_history_invalid_indices(self, mock_load):
        """Test deleting with invalid indices"""
        mock_load.return_value = self.mock_history.copy()
        
        with patch('builtins.print'):
            result = delete_selected_history([10, 20])  # Invalid indices
            self.assertFalse(result)

    @patch('utils.load_analysis_history')
    def test_get_history_count(self, mock_load):
        """Test getting history count"""
        mock_load.return_value = self.mock_history
        
        count = get_history_count()
        self.assertEqual(count, 2)
        
        # Test with empty history
        mock_load.return_value = []
        count_empty = get_history_count()
        self.assertEqual(count_empty, 0)

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.rmdir')
    @patch('builtins.print')
    def test_clear_empty_thumbnail_folder_success(self, mock_print, mock_rmdir, 
                                                 mock_listdir, mock_exists):
        """Test clearing empty thumbnail folder"""
        mock_exists.return_value = True
        mock_listdir.return_value = []  # Empty folder
        
        clear_empty_thumbnail_folder()
        
        mock_rmdir.assert_called_once()

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.print')
    def test_clear_empty_thumbnail_folder_not_empty(self, mock_print, mock_listdir, mock_exists):
        """Test clearing non-empty thumbnail folder"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['thumb1.jpg']  # Not empty
        
        with patch('os.rmdir') as mock_rmdir:
            clear_empty_thumbnail_folder()
            mock_rmdir.assert_not_called()

    @patch('os.path.exists')
    def test_clear_empty_thumbnail_folder_not_exists(self, mock_exists):
        """Test clearing thumbnail folder that doesn't exist"""
        mock_exists.return_value = False
        
        with patch('os.rmdir') as mock_rmdir:
            clear_empty_thumbnail_folder()
            mock_rmdir.assert_not_called()

    def test_constants(self):
        """Test that constants are properly defined"""
        self.assertIsInstance(HISTORY_FILE, str)
        self.assertIsInstance(THUMBNAIL_DIR, str)
        self.assertTrue(len(HISTORY_FILE) > 0)
        self.assertTrue(len(THUMBNAIL_DIR) > 0)

if __name__ == '__main__':
    unittest.main()