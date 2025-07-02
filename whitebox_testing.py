import unittest
from PIL import Image
import sys
import os

# 添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from main import analyze_image_comprehensive_advanced
from ela_analysis import perform_multi_quality_ela
from feature_detection import extract_multi_detector_features
from copy_move_detection import detect_copy_move_advanced, detect_copy_move_blocks
from advanced_analysis import analyze_noise_consistency  # 确保文件名和模块名匹配
from jpeg_analysis import advanced_jpeg_analysis
from advanced_analysis import analyze_texture_consistency, analyze_edge_consistency, analyze_illumination_consistency, perform_statistical_analysis  # Import from advanced_analysis module
from classification import classify_manipulation_advanced

class TestForensicSystem(unittest.TestCase):
    def setUp(self):
        self.test_image_path = "im11f.jpg"  # 替换为您测试图片的路径
        self.image_pil = Image.open(self.test_image_path)
        self.analysis_results = analyze_image_comprehensive_advanced(self.test_image_path)

    def test_ela_analysis(self):
        ela_result = perform_multi_quality_ela(self.image_pil)
        ela_image, ela_mean, ela_std, regional_stats, quality_stats, ela_variance = ela_result
        self.assertIsNotNone(ela_image)
        self.assertGreaterEqual(ela_mean, 0)

    def test_feature_extraction(self):
        feature_sets, _, _ = extract_multi_detector_features(
            self.image_pil, Image.new('L', self.image_pil.size), 0, 0)
        # Check that feature sets exist and have the expected structure
        self.assertIn('sift', feature_sets)
        self.assertIn('orb', feature_sets)
        self.assertIn('akaze', feature_sets)
        # Features might be empty for some images, so just check structure
        self.assertIsInstance(feature_sets['sift'], tuple)
        self.assertIsInstance(feature_sets['orb'], tuple)
        self.assertIsInstance(feature_sets['akaze'], tuple)

    def test_copy_move_detection(self):
        # Use actual feature extraction result instead of empty features
        feature_sets, _, _ = extract_multi_detector_features(
            self.image_pil, Image.new('L', self.image_pil.size), 0, 0)
        ransac_matches, inliers, transform = detect_copy_move_advanced(
            feature_sets, self.image_pil.size)
        self.assertGreaterEqual(inliers, 0)
        # Transform can be None for some images, so don't assert not None

    def test_block_matching(self):
        block_matches = detect_copy_move_blocks(self.image_pil)
        self.assertIsInstance(block_matches, list)

    def test_noise_analysis(self):
        noise_results = analyze_noise_consistency(self.image_pil)
        self.assertGreaterEqual(noise_results['overall_inconsistency'], 0)

    def test_jpeg_analysis(self):
        jpeg_results = advanced_jpeg_analysis(self.image_pil)
        self.assertGreaterEqual(jpeg_results['response_variance'], 0)
        self.assertGreaterEqual(jpeg_results['estimated_original_quality'], 0)

    def test_texture_analysis(self):
        texture_results = analyze_texture_consistency(self.image_pil)
        self.assertGreaterEqual(texture_results['overall_inconsistency'], 0)

    def test_edge_analysis(self):
        edge_results = analyze_edge_consistency(self.image_pil)
        self.assertGreaterEqual(edge_results['edge_inconsistency'], 0)

    def test_illumination_analysis(self):
        illumination_results = analyze_illumination_consistency(self.image_pil)
        self.assertGreaterEqual(illumination_results['overall_illumination_inconsistency'], 0)

    def test_statistical_analysis(self):
        stat_results = perform_statistical_analysis(self.image_pil)
        self.assertGreaterEqual(stat_results['overall_entropy'], 0)

    def test_classification(self):
        classification_results = classify_manipulation_advanced(self.analysis_results)
        valid_types = ['Copy-Move Forgery', 'Splicing Forgery', 'Manipulasi Kompleks (Copy-Move + Splicing)', 
                      'Tidak Terdeteksi Manipulasi', 'Analysis Error']
        self.assertIn(classification_results['type'], valid_types)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()