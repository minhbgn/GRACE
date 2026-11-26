# GRACE recovery investigation — 2026-10-27

The original reproduction/GRACE directory is empty. No original project files were changed during this investigation. Recovery artifacts were saved to /tmp, a separate filesystem from /raid.

## Evidence

- VS Code Git log opened GRACE successfully at 11:54:15 Singapore time on 2026-10-27.
- At 11:57:24, that log reported missing GRACE/.git/HEAD and .git/config.
- The empty GRACE directory had modification time 12:36:48. This is not proof of the deletion time or cause.
- Parent Git index contains mode 160000 (gitlink) for reproduction/GRACE, commit b73fa9b4d427aa64ed3ad274def641efb3145877. This stores a nested-repository commit reference, not its file contents. The referenced commit object is absent from parent Git; there is no .gitmodules or .git/modules backup.
- Server user trash at ~/.local/share/Trash is absent. /raid/.Trash-1000 exists but contains no GRACE folder. No matching copy was found in the searched nearby directories. This does not exclude external backups, client-side trash, or administrator snapshots.
- Cause is unconfirmed. The inspected logs do not identify the deleting operation.

## Preserved material

- literal_snapshots/: 67 distinct paths extracted as literal file bodies from recorded quoted heredocs or add-file patches; 69 versions retained under versions/. These are historical snapshots and can precede subsequent edits. They are not a verified reconstruction of the final working tree.
- editor_history/: 19 saved editor versions, with provenance in editor_provenance.json.
- historical_commands.jsonl: 1,830 historical tool command records mentioning GRACE, with source session and line. These are evidence only. No historical commands were executed. Further reconstruction can use these records.
- temporary_artifacts/ and temporary_backups/: surviving scripts, reports, validation output, plans, and intermediate code backups.
- evidence/: VS Code logs and a copy of the parent Git index.

The four currently open experiment files have historical copies under literal_snapshots/experiments/qat_int4_token_reduction_576_seed42/. These snapshots have not been installed into the original project or validated as runnable.

## Remaining recovery

1. Keep open editor buffers available; save any surviving contents to a separate filesystem.
2. Check client Recycle Bin/Trash if applicable, and ask the server administrator for a backup or snapshot from before 11:54 on 2026-10-27.
3. Full checkpoints, datasets, environments, and complete experiment results have not been recovered by this extraction. Re-cloning the public repository restores upstream source only, not local research changes or trained weights.
4. If backups are unavailable and deleted binary artifacts matter, minimize writes to /raid and ask the administrator to assess filesystem recovery before downloading/rebuilding there.
