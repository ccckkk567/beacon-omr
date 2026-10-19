"""Compact request preprocessing and original-slot scatter, engine RPC unchanged."""
import time
import numpy as np
import torch
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto


def scatter_slots(compact, indices, size):
    tensors = {}
    for key, value in compact.batch.items():
        full = torch.zeros((size, *value.shape[1:]), dtype=value.dtype, device=value.device)
        full[torch.as_tensor(indices, device=value.device)] = value
        tensors[key] = full
    non_tensors = {}
    for key, value in compact.non_tensor_batch.items():
        if value.dtype == object:
            full = np.empty((size, *value.shape[1:]), dtype=object)
            full.fill(None)
        else:
            full = np.zeros((size, *value.shape[1:]), dtype=value.dtype)
        if key in ('omr_synced_policy_version', 'omr_weight_sync_sequence'):
            full.fill(-1)  # Explicitly not an engine receipt for inactive rows.
        full[indices] = value
        non_tensors[key] = full
    generated = np.zeros(size, dtype=bool)
    generated[indices] = True
    non_tensors['has_generated_response'] = generated
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=compact.meta_info)


def generate_turn(collector, gen_batch, observations, indices, seeds, actor_rollout_wg,
                  observer=None, local_step=0):
    """Exactly one ordinary generate_sequences RPC per nonempty driver turn."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return None
    if observer is not None:
        observer.phase('preprocess', local_step)
    started = time.monotonic()
    compact_obs = {key: None if value is None else [value[int(i)] for i in indices]
                   for key, value in observations.items()}
    batch = collector.preprocess_batch(gen_batch.select_idxs(indices), compact_obs)
    if 'index' in batch.non_tensor_batch:
        batch.non_tensor_batch['index'] = indices.copy()
    keys = [key for key in ('raw_prompt_ids', 'multi_modal_data', 'raw_prompt', 'tools_kwargs')
            if key in batch.non_tensor_batch]
    inputs = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                       non_tensor_batch_keys=keys)
    inputs.meta_info = dict(gen_batch.meta_info)
    if observer is not None:
        inputs.meta_info['omr_expected_policy_version'] = observer.update
        inputs.non_tensor_batch['omr_generation_seed'] = np.asarray(seeds, dtype=np.int64)[indices]
    padded, pad_size = pad_dataproto_to_divisor(inputs, actor_rollout_wg.world_size)
    preprocess_wall = time.monotonic() - started
    if observer is not None:
        observer.phase('generate', local_step)
    started = time.monotonic()
    output = actor_rollout_wg.generate_sequences(padded)
    generation_wall = time.monotonic() - started
    if observer is not None:
        # Before env.step, including DP padding and legacy terminal requests.
        observer.on_generation_completed(padded, output, indices, pad_size, local_step,
                                         preprocess_wall, generation_wall)
    output = unpad_dataproto(output, pad_size)
    batch = batch.union(output)
    batch.non_tensor_batch['generation_compact_index'] = np.arange(len(indices))
    return scatter_slots(batch, indices, len(gen_batch))
