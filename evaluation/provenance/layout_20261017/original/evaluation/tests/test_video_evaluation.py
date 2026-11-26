"""Input preservation, actual processor accounting, offline assets and integrity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_protocol import ROOT, geometry, input_counts, inventory, local_video, sampling
from run_video_matrix import summary_from_rows
from local_scoring import generation_record


class VideoPolicyTests(unittest.TestCase):
    def test_sampling_keeps_long_clips_and_endpoints(self):
        indices = sampling(18000, 30, 2)
        self.assertEqual(len(indices), 1200)  # exceeds Qwen's native 768-frame cap
        self.assertEqual((indices[0], indices[-1]), (0, 17999))
        self.assertEqual(len(set(indices)), len(indices))
        self.assertEqual(sampling(3, 24, 0), [0, 1, 2])
        self.assertEqual(sampling(1, 24, 2), [0])

    def test_archive_is_used_without_bulk_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / 'chunk.zip', 'w') as archive:
                archive.writestr('data/example.mp4', b'video payload')
                archive.writestr('data/unused.mp4', b'not needed')
            assets = inventory(root)
            with local_video(assets['example'], root) as path:
                self.assertEqual(Path(path).read_bytes(), b'video payload')
                extracted = Path(path)
            self.assertFalse(extracted.exists())
            self.assertFalse((root / 'data').exists())

    def test_source_geometry_is_independent_of_duration_budgets(self):
        short = geometry(14, 316, 600, 16, 2, 2)
        long = geometry(1400, 316, 600, 16, 2, 2)
        self.assertEqual((short['height'], short['width']), (long['height'], long['width']))
        self.assertEqual(long['visual_tokens'], short['visual_tokens'] * 100)


class IntegrityTests(unittest.TestCase):
    def record(self):
        return dict(id='one', gold='B', raw='B', options={'A': 'left', 'B': 'right'},
                    prediction='B', correct=True, question_type='multiple-choice', category='action',
                    input_identity='input hash', eos_token_id=99,
                    generation=generation_record([2, 99], 99, 4096),
                    tokens=dict(input_visual_tokens=1000, input_text_tokens=80,
                                input_tokens=1080, output_text_tokens=2))

    def test_scores_and_token_totals_recomputed(self):
        row = self.record()
        result = summary_from_rows([row])
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['token_usage']['total_text_tokens']['mean'], 82)
        row['correct'] = False
        with self.assertRaisesRegex(ValueError, 'score'):
            summary_from_rows([row])

    def test_censored_outputs_are_visible(self):
        row = self.record()
        row['generation'] = generation_record([2, 3], 99, 2)
        self.assertEqual(summary_from_rows([row])['generation_cutoffs'], 1)
        row['tokens']['input_tokens'] = 1079
        with self.assertRaisesRegex(ValueError, 'accounting'):
            summary_from_rows([row])


class RealProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = ROOT / 'models/Qwen3-VL-2B-Instruct'
        if not model.is_dir():
            raise unittest.SkipTest('Local processor assets unavailable')
        from transformers import AutoProcessor
        cls.processor = AutoProcessor.from_pretrained(model, local_files_only=True)

    def test_odd_frame_padding_and_exact_prompt_tokens(self):
        import torch
        from transformers.video_utils import VideoMetadata
        from video_protocol import prompt_text
        processor = self.processor
        row = {'prompt': 'What happens in the video? Answer briefly.'}
        for n in (1, 3, 8):
            with self.subTest(frames=n):
                shape = geometry(n, 64, 96, 16, 2, 2)
                meta = dict(frame_indices=list(range(n)), source_fps=10, **shape)
                before = copy.deepcopy(meta)
                expected = input_counts(processor, row, meta)
                self.assertEqual(meta, before)  # HF timestamp padding mutates its argument
                processor.video_processor.size = {'shortest_edge': 1, 'longest_edge': shape['total_pixels'] + 1}
                actual = processor(text=[prompt_text(processor, row)],
                    videos=[torch.zeros(n, 3, 64, 96, dtype=torch.uint8)],
                    videos_kwargs=dict(do_resize=False, do_sample_frames=False,
                        video_metadata=[VideoMetadata(total_num_frames=n, fps=10, frames_indices=list(range(n)))]),
                    truncation=False, return_tensors='pt')
                ids = actual['input_ids'][0]
                visual = int((ids == processor.video_token_id).sum())
                self.assertEqual(expected, dict(input_visual_tokens=visual,
                    input_text_tokens=len(ids) - visual, input_tokens=len(ids)))
                self.assertEqual(actual['video_grid_thw'][0].tolist(), shape['grid'])


if __name__ == '__main__':
    unittest.main()
