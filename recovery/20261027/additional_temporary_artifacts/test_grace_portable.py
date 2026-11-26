import json
import os
from pathlib import Path
import sys
import torch
from torch import nn
from transformers import Trainer, TrainingArguments, TrainerCallback
sys.path.insert(0, str(Path.cwd() / 'reproduction/GRACE/qwen-vl-finetune'))
from qwenvl.train.grace_resume import CheckpointPauseCallback

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, input_ids, labels):
        logits = self.linear(input_ids)
        return {'loss': ((logits-labels)**2).mean(), 'logits': logits}

class SaveIB(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            (Path(args.output_dir)/f'checkpoint-{state.global_step}'/'grace_ib_state.json').write_text('{}')

output = sys.argv[1]
resume = sys.argv[2] == 'resume'
world = int(os.environ['WORLD_SIZE'])
args = TrainingArguments(output_dir=output, use_cpu=True, ddp_backend='gloo',
    per_device_train_batch_size=1, gradient_accumulation_steps=8//world,
    max_steps=4, save_steps=2, save_total_limit=5, report_to='none',
    disable_tqdm=True, dataloader_drop_last=True)
callback = CheckpointPauseCallback()
trainer = Trainer(model=Model(), args=args, train_dataset=[{'input_ids': torch.tensor([1., 2.]), 'labels': torch.tensor([0.])} for _ in range(128)], callbacks=[SaveIB(), callback])
trainer.train(resume_from_checkpoint=True if resume else None)
assert trainer.state.global_step == (4 if resume else 2)
assert callback.paused == (not resume)
if resume:
    assert max(float(v['step']) for v in trainer.optimizer.state.values()) == 4
if trainer.is_world_process_zero():
    print(f'PASS world={world}, step={trainer.state.global_step}, paused={callback.paused}', flush=True)
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
