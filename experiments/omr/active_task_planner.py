"""Outer fixed-budget task planning; the checkpoint-level mixture is unchanged."""
from copy import deepcopy
import random
from .demo_manifest import PROFILE, TASK_SEED
from .task_manifest import digest


class ActiveTaskPlanner:
    def __init__(self, tasks, branch_masks, root_schedule_hash):
        self.tasks = {row['id']: deepcopy(row) for row in tasks}
        self.masks = deepcopy(branch_masks)
        self.binding = dict(profile=PROFILE, membership=digest(self.tasks),
                            branch_masks=digest(branch_masks), root_schedule=root_schedule_hash)
        self.rng = random.Random(TASK_SEED)
        self.cursor = 0

    def plan(self, root_groups, frozen_entries, update):
        if update != self.cursor or len(root_groups) != 16:
            raise ValueError('Active task planner cursor/budget mismatch')
        for row in root_groups:
            if self.tasks.get(row['id']) != row:
                raise ValueError('Root bank escaped the demo pool')
        eligible = sorted({e.task.instance_id for e in frozen_entries.values()
                           if e.validity_status == 'verified_candidate'})
        if set(eligible) - self.tasks.keys():
            raise ValueError('Archive contains a task outside demo training membership')
        result, available, cycle = [], [], -1
        for slot, root in enumerate(root_groups):
            planned_archive = slot in self.masks[update]
            chosen = root
            remaining = []
            if planned_archive and eligible:
                if not available:
                    available = list(eligible)
                    self.rng.shuffle(available)
                    cycle += 1
                remaining = sorted(available)
                chosen = self.tasks[available.pop()]
            result.append(dict(slot=slot, root_bank_task_id=root['id'], task=deepcopy(chosen),
                planned_branch='archive' if planned_archive else 'root',
                actual_request_branch='archive' if planned_archive and eligible else 'root',
                empty_fallback=planned_archive and not eligible,
                eligible_task_count=len(eligible), conditional_task_set=remaining,
                task_conditional_probability=1/len(remaining) if remaining else 1.,
                cycle=cycle if remaining else None, cycle_position=len(eligible)-len(remaining) if remaining else None,
                branch_slot_fraction=.5, branch_assignment_probability=1.,
                update=update, root_schedule_sha256=self.binding['root_schedule'],
                branch_masks_sha256=self.binding['branch_masks']))
        self.cursor += 1
        return result

    def state_dict(self):
        return dict(binding=self.binding, cursor=self.cursor, rng=self.rng.getstate())

    def load_state_dict(self, state):
        if state['binding'] != self.binding or not 0 <= state['cursor'] <= 150:
            raise ValueError('Active planner checkpoint binding differs')
        def tuples(x):
            return tuple(map(tuples, x)) if isinstance(x, (list, tuple)) else x
        self.rng.setstate(tuples(state['rng']))
        self.cursor = state['cursor']


def metrics(rows, actual_ids):
    requested = [r for r in rows if r['actual_request_branch'] == 'archive']
    actual = sum(key is not None for key in actual_ids)
    if len(rows) != 16 or len(actual_ids) != 16 or actual > len(requested):
        raise ValueError('Invalid fixed group budget')
    result = {'demo/planned_root_groups': 8, 'demo/planned_archive_groups': 8,
              'demo/actual_root_groups': 16-actual, 'demo/actual_archive_groups': actual,
              'demo/archive_request_groups': len(requested), 'demo/actual_restart_fraction': actual/16,
              'demo/eligible_archive_tasks': rows[0]['eligible_task_count'],
              'demo/empty_fallback_groups': sum(r['empty_fallback'] for r in rows),
              'demo/restore_failed_groups': len(requested)-actual,
              'demo/distinct_archive_tasks': len({r['task']['id'] for r in requested}),
              'demo/repeated_archive_task_groups': len(requested)-len({r['task']['id'] for r in requested})}
    if requested:
        result['demo/restore_success_rate'] = actual/len(requested)
    return result
