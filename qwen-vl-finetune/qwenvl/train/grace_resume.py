"""Collective checkpoint-boundary pause, after model/optimizer and IB persistence."""
import os
from pathlib import Path
import shutil

import torch
import torch.distributed as dist
from transformers import TrainerCallback


class CheckpointPauseCallback(TrainerCallback):
    def __init__(self):
        self.paused = False

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            (Path(args.output_dir) / 'PAUSED_AT_CHECKPOINT').unlink(missing_ok=True)

    def on_save(self, args, state, control, **kwargs):
        output = Path(args.output_dir)
        checkpoint = output / f'checkpoint-{state.global_step}'
        # Earlier callbacks have saved processor and adaptive IB controller.
        # Wait for all ranks to finish checkpoint writes before acknowledging pause.
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            dist.barrier()
        stop = False
        if state.is_world_process_zero:
            recipe = os.environ.get('GRACE_LAUNCH_RECIPE')
            if recipe:
                shutil.copyfile(recipe, checkpoint / 'launch_recipe.json')
            stop = (output / 'STOP_AFTER_CHECKPOINT').exists()
        if distributed:
            flag = torch.tensor(int(stop), device=args.device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            self.paused = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                (output / 'STOP_AFTER_CHECKPOINT').unlink(missing_ok=True)
                (output / 'PAUSED_AT_CHECKPOINT').write_text(str(checkpoint) + '\n')
        return control
