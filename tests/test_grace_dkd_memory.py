"""CPU checks for the memory-safe DKD path; no models or GPUs required."""
import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "qwen-vl-finetune"))
from qwenvl.train.grace_modules import gated_dkd_loss


class CheckpointedDKDTests(unittest.TestCase):
    def test_loss_and_gradients_match(self):
        torch.manual_seed(12)
        for dtype in (torch.float32, torch.bfloat16):
            for batch in (1, 2):
                for chunk in (1, 4, 32):
                    with self.subTest(dtype=dtype, batch=batch, chunk=chunk):
                        # Noncontiguous sequence views and mixed ignored labels.
                        source = torch.randn(batch, 22, 29, dtype=dtype)[:, ::2]
                        teacher = torch.randn_like(source)
                        labels = torch.randint(0, 29, (batch, 11))
                        labels[:, 2:5] = -100
                        reference = source.detach().requires_grad_()
                        actual = source.detach().requires_grad_()
                        old = gated_dkd_loss(reference, teacher, labels, chunk_size=chunk)
                        new = gated_dkd_loss(actual, teacher, labels, chunk_size=chunk,
                                             checkpoint_chunks=True)
                        old.backward()
                        new.backward()
                        tol = 2e-2 if dtype == torch.bfloat16 else 2e-5
                        torch.testing.assert_close(new, old, atol=tol, rtol=tol)
                        torch.testing.assert_close(actual.grad, reference.grad,
                                                   atol=tol, rtol=tol)
                        self.assertIsNone(teacher.grad)

    def test_empty_supervision_and_inference(self):
        student = torch.randn(2, 7, 13)
        teacher = torch.randn_like(student)
        labels = torch.full((2, 7), -100)
        loss = gated_dkd_loss(student, teacher, labels, checkpoint_chunks=True)
        self.assertEqual(loss.item(), 0)
        labels[:, 3:] = 2
        with torch.no_grad():
            torch.testing.assert_close(
                gated_dkd_loss(student, teacher, labels),
                gated_dkd_loss(student, teacher, labels, chunk_size=2,
                               checkpoint_chunks=True),
            )

    def test_no_retained_vocabulary_intermediates(self):
        student = torch.randn(1, 65, 127, requires_grad=True)
        teacher = torch.randn_like(student)
        labels = torch.randint(0, 127, (1, 65))
        original_storages = {x.untyped_storage().data_ptr() for x in (student, teacher)}

        def retained_bytes(checkpointed):
            saved = {}

            def pack(tensor):
                storage = tensor.untyped_storage()
                if tensor.ndim >= 2 and tensor.shape[-1] == 127:
                    if storage.data_ptr() not in original_storages:
                        saved[storage.data_ptr()] = storage.nbytes()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                loss = gated_dkd_loss(student, teacher, labels, chunk_size=8,
                                      checkpoint_chunks=checkpointed)
            total = sum(saved.values())
            del loss
            return total

        self.assertGreater(retained_bytes(False), 0)
        self.assertEqual(retained_bytes(True), 0)

    def test_invalid_chunk_size(self):
        for checkpointed in (False, True):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                gated_dkd_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3),
                               torch.zeros(1, 2, dtype=torch.long), chunk_size=0,
                               checkpoint_chunks=checkpointed)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
