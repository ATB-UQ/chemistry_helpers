Captured read-only 2026-10-20 (`ssh helium`, `ssh setonix`), one batched ssh per host, each
with a consumer's exact squeue format. Job names are the account's own (`rl57_...`).
`setonix_login_banner_only.txt` is the real stdout of `squeue -u $USER -h -t all -o '%i %T'`
on an empty Setonix queue: the Pawsey NOTICE banner and no job rows.
`helium_narrow_8T.txt` is `--format='%8T %i'`: COMPLETED printed as COMPLETE (the width trap).
