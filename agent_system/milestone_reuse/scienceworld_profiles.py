"""Versioned OMR questions; BEACON supplies the native milestone detector."""
SIGNAL_PROFILE = 'scienceworld-beacon-boundary-v1'
STATISTICS_PROFILE = 'beacon-progress-after-start-v1'
NOVELTY_PROFILE = 'beacon-boundary-condition-v1'
PROGRESS_EVENT = 'sw_beacon_progress_after_start_v1'
EVENT_IDS = (PROGRESS_EVENT, 'Goal')
PROFILES = dict(signal_profile_id=SIGNAL_PROFILE,
                statistics_profile_id=STATISTICS_PROFILE, novelty_profile_id=NOVELTY_PROFILE)
