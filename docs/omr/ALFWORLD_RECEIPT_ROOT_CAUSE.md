# ALFWorld Full receipt repair, 2026-10-19

The supplied repair plan reports Full attempt 167234-3 at source 98d7a69 failing
before its first update with `Explicit replay loaded a different ALFWorld game`.
The original log is referenced by SHA256
`29eb4cd1d3b62bbc7fc82be64cd364bd5138532b85d29f177998a6bd83d4020a` in that plan;
it has not been independently recovered in this checkout. No original raw
gamefile field or imported HPC wrapper path has yet been supplied.

Source mechanism: AlfredInfos wrote extra.gamefile only on reset; a fresh step
state can lose it. Filter may return a present None, for which dict.get's default
does not apply. NativeReplay then mistook unavailable identity for a different
trial. Baseline does not make this per-step receipt check.

The repaired wrapper binds identity to the file actually passed to successful
native load, propagates it on reset and step, and rejects an explicit conflicting
native identity. It does not bind to a requested task as a fallback. Receipts
validate normalized full path and trial ID, with unavailable/mismatch/shape
errors separated. Reset epoch and local action/step diagnostics are additive.
Observation, reward, done and admissible commands are retained.

CPU fixtures exercise the production wrapper with fresh/reset-only states,
two resets, explicit conflicting paths, malformed batches, and replay plus
independent continuation. These are not native TextWorld/GPU acceptance.
The repaired formal Full allocation runs exactly one independent simulator
reset/step/capture/replay/continuation check before validation. Its actual
versions, source files and raw fields are saved in
`attempts/<attempt>/alfworld-native-identity.json`. Native acceptance and the
first optimizer update remain pending until those execution receipts return.

Normal baseline and all other environments are protected. No engine lifecycle,
model/reward/horizon/batch/profile change is part of this identity repair.
