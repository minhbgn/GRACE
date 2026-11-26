"""Durable, identity-checked answer prefixes for the video harness adapter."""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def records(path):
    if not path.exists():
        return []
    result = []
    with path.open() as stream:
        for line in stream:
            # Only a torn final append is recoverable. Complete corrupt rows fail closed.
            if not line.endswith('\n'):
                if stream.read():
                    raise ValueError(f'Corrupt checkpoint: {path}')
                break
            value = json.loads(line)
            checksum = value.pop('checksum')
            if checksum != digest(value) or value['index'] != len(result):
                raise ValueError(f'Checkpoint checksum/order mismatch: {path}')
            result.append(value)
    return result


class AnswerCheckpoint:
    def __init__(self, attempt, identity):
        self.attempt, self.identity = Path(attempt), identity
        self.prefix = []
        self.index = 0
        for old in sorted(self.attempt.parent.glob('attempt_*')):
            if old == self.attempt:
                continue
            rows = records(old / 'answers.checkpoint.jsonl')
            if rows and any(row['identity'] != identity for row in rows):
                raise ValueError(f'Checkpoint provenance changed: {old}')
            if len(rows) > len(self.prefix):
                self.prefix = rows
        print(f'RESUME: {len(self.prefix)} durable answers available', flush=True)

    def replay(self, key):
        if self.index >= len(self.prefix):
            return None
        row = self.prefix[self.index]
        if row['key'] != key or row['telemetry']['index'] != self.index:
            raise ValueError('Checkpoint request identity/order changed')
        append(self.attempt / 'telemetry.jsonl', row['telemetry'])
        append(self.attempt / 'video_inputs.jsonl', row['video_input'])
        self.save(key, row['answer'], row['telemetry'], row['video_input'])
        return row['answer']

    def save(self, key, answer, telemetry, video_input):
        if telemetry['index'] != self.index:
            raise ValueError('Checkpoint telemetry index mismatch')
        row = dict(identity=self.identity, index=self.index, key=key, answer=answer,
                   telemetry=telemetry, video_input=video_input)
        append(self.attempt / 'answers.checkpoint.jsonl', dict(row, checksum=digest(row)))
        self.index += 1
