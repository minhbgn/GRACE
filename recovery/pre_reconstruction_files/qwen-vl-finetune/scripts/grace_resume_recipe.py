"""Validate the resolved launcher recipe before loading models or optimizer shards."""
import json
from pathlib import Path
import sys


def prepare(output, world_size, effective_batch, ds_config, pairs):
    output = Path(output)
    recipe = dict(pair.split('=', 1) for pair in pairs)
    recipe['effective_batch'] = int(effective_batch)
    recipe['world_size'] = int(world_size)
    if recipe['TRAINING_BACKEND'] == 'zero2':
        recipe['deepspeed_config'] = json.loads(Path(ds_config).read_text())
    checkpoints = sorted(
        (p for p in output.glob('checkpoint-*') if p.is_dir() and p.name[11:].isdigit()),
        key=lambda p: int(p.name[11:]),
    )
    if checkpoints:
        latest = checkpoints[-1]
        manifest = latest / 'launch_recipe.json'
        if not manifest.exists():
            raise SystemExit(
                f'ERROR: {latest} predates resume recipe tracking. Use a fresh OUTPUT_DIR '
                'for the portable DDP run; legacy checkpoints require explicit migration.'
            )
        previous = json.loads(manifest.read_text())
        ignored = {'world_size'} if recipe['TRAINING_BACKEND'] == 'ddp' else set()
        changed = [k for k in previous.keys() | recipe.keys()
                   if k not in ignored and previous.get(k) != recipe.get(k)]
        if changed:
            raise SystemExit('ERROR: incompatible resume recipe: ' + ', '.join(sorted(changed)))
        if not (latest / 'trainer_state.json').exists() or not (latest / 'grace_ib_state.json').exists():
            raise SystemExit(f'ERROR: incomplete checkpoint: {latest}')
        if recipe['TRAINING_BACKEND'] == 'ddp':
            for name in ('optimizer.pt', 'scheduler.pt'):
                if not (latest / name).exists():
                    raise SystemExit(f'ERROR: missing resumable state: {latest / name}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'launch_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')


if __name__ == '__main__':
    prepare(*sys.argv[1:5], sys.argv[5:])
