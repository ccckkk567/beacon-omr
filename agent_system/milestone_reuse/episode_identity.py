"""Strict ALF identity contracts, independent of simulator/learner imports."""
import json
import os

from .identity import alfworld_trial_id


class IdentityContractError(ValueError):
    def __init__(self, code, **evidence):
        self.code, self.evidence = code, evidence
        super().__init__(code + ': ' + json.dumps(evidence, ensure_ascii=False, default=repr))


def gamefile_value(raw, **evidence):
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise IdentityContractError('identity_shape_error', raw_type=type(raw).__name__,
                                        raw_value=raw, **evidence)
        raw = raw[0]
    if raw is None:
        raise IdentityContractError('identity_unavailable', raw_type='NoneType', **evidence)
    if not isinstance(raw, (str, os.PathLike)) or not os.fspath(raw):
        raise IdentityContractError('identity_shape_error', raw_type=type(raw).__name__,
                                    raw_value=raw, **evidence)
    value = os.fspath(raw)
    if not isinstance(value, str):
        raise IdentityContractError('identity_shape_error', raw_type=type(value).__name__, **evidence)
    return value


def verify_gamefile(raw, expected, *, task_id=None, **evidence):
    actual = gamefile_value(raw, **evidence)
    expected = gamefile_value(expected)
    # Full normalized file identity AND the manifest trial, never basename alone.
    normalize = lambda p: os.path.normpath(p.replace('\\', '/')).replace('\\', '/')
    try:
        trial = alfworld_trial_id(actual)
    except ValueError as error:
        raise IdentityContractError('identity_shape_error', actual=actual, **evidence) from error
    if normalize(actual) != normalize(expected) or (task_id is not None and trial != task_id):
        raise IdentityContractError('identity_mismatch', actual=actual, requested=expected,
                                    requested_task_id=task_id, **evidence)
    return actual
