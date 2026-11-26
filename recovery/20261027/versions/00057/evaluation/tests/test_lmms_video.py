"""Protocol regression checks without model weights or harness imports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import unittest
from evaluation.protocols.video_protocol import geometry
from evaluation.runners.run_lmms_matrix import VIDEO_TASKS, video_local_choice, validate_result, video_no_network

class VideoProtocolTests(unittest.TestCase):
    def test_720p_geometry_has_more_tokens_and_does_not_upscale(self):
        old = geometry(20, 1080, 1920, 16, 2, 2, 256)
        new = geometry(20, 1080, 1920, 16, 2, 2, 1280, 720)
        self.assertEqual((new['height'], new['width']), (704, 1280))
        self.assertGreater(new['visual_tokens'], old['visual_tokens'])
        small = geometry(20, 320, 640, 16, 2, 2, 1280, 720)
        self.assertEqual((small['height'], small['width']), (320, 640))
        portrait = geometry(20, 1920, 1080, 16, 2, 2, 1280, 720)
        self.assertEqual((portrait['height'], portrait['width']), (1280, 704))

    def test_tempcompass_fallback_counts_unresolved_wrong(self):
        doc = dict(video_id='v', dim='action', _grace_row=json.dumps({
            'options': {'A': 'left', 'B': 'right'}, 'gold': 'B'}))
        self.assertEqual(video_local_choice(doc, ['B'])['avg_accuracy']['rating'], 1)
        result = video_local_choice(doc, ['I cannot determine this.'])['avg_accuracy']
        self.assertEqual(result['rating'], 0)
        self.assertTrue(result['unresolved'])

    def test_network_is_forbidden_even_with_credentials(self):
        import requests
        with video_no_network(), self.assertRaisesRegex(RuntimeError, 'Network forbidden'):
            requests.post('https://example.invalid/judge', headers={'Authorization': 'Bearer fake'})

    def test_video_score_scales_and_smoke_coverage(self):
        for suite, (task, _, metric, scale) in VIDEO_TASKS.items():
            results = {'results': {task: {metric: 75 / scale}},
                       'n-samples': {task: {'original': 50, 'effective': 2}}, 'config': {'limit': 2}}
            samples = [dict(doc_id=i, doc_hash=str(i), target='A', resps=[['A']]) for i in range(2)]
            self.assertEqual(validate_result(results, samples, suite, 2, VIDEO_TASKS)['score_percent'], 75)
            with self.assertRaises(ValueError):
                validate_result(results, samples, suite, None, VIDEO_TASKS)

if __name__ == '__main__':
    unittest.main()
