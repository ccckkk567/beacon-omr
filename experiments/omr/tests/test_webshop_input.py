from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent_system.environments.webshop_input import fit_input, chat_text, INPUT_PROFILE, WebshopInputBudgetError
from agent_system.milestone_reuse.native_restart import render_starts
from experiments.omr.tests.test_full_protocol import config_for, native_manager


class CharacterTokenizer:
    def apply_chat_template(self, chat, **kw):
        return '<user>' + chat[0]['content'] + '</user><assistant>'
    def __call__(self, text, **kw):
        return {'input_ids': [ord(c) for c in text]}
    def decode(self, ids, **kw):
        return ''.join(chr(c) for c in ids)


class WebshopInputTests(unittest.TestCase):
    def fit(self, **kw):
        base = dict(task='buy blue', observation='current real page', actions="'click[blue]',\n'click[Buy Now]',",
                    history='old observation and action', history_length=2, step_count=3,
                    use_history=True, tokenizer=CharacterTokenizer())
        return fit_input(**dict(base, **kw))

    def test_short_input_is_unchanged_and_actual_chat_tokens_are_measured(self):
        text, audit = self.fit()
        self.assertFalse(audit['history_removed'])
        self.assertFalse(audit['observation_trimmed'])
        self.assertNotIn('original_input', audit)
        self.assertEqual(audit['actual_tokens'], len(chat_text(CharacterTokenizer(), text)))
        self.assertIn('old observation and action', text)

    def test_history_overflow_falls_back_before_trimming_current_page(self):
        text, audit = self.fit(history='old history ' * 1500)
        self.assertTrue(audit['history_removed'])
        self.assertFalse(audit['observation_trimmed'])
        self.assertGreater(audit['original_tokens'], 7000)
        self.assertLessEqual(audit['actual_tokens'], 7000)
        self.assertIn('current real page', text)
        self.assertIn('click[Buy Now]', text)
        self.assertNotIn('old history', text)

    def test_character_threshold_applies_even_if_tokens_fit(self):
        class Compressed(CharacterTokenizer):
            def __call__(self, text, **kw):
                return {'input_ids': text.split()}
        text, audit = self.fit(history='abcdefghij ' * 1300, tokenizer=Compressed())
        self.assertGreater(audit['original_chars'], 13000)
        self.assertLess(audit['original_tokens'], 7000)
        self.assertTrue(audit['history_removed'])

    def test_token_threshold_also_applies_below_character_threshold(self):
        _, audit = self.fit(history='x' * 7000)
        self.assertLess(audit['original_chars'], 13000)
        self.assertTrue(audit['history_removed'])

    def test_only_observation_is_trimmed_and_immutable_parts_are_preserved(self):
        for body in ('page line\n' * 3000, '商品' * 10000):
            with self.subTest(body=body[:12]):
                text, audit = self.fit(observation=body)
                self.assertTrue(audit['observation_trimmed'])
                self.assertLessEqual(audit['actual_tokens'], 7000)
                self.assertIn('buy blue', text)
                self.assertIn('click[Buy Now]', text)
                self.assertIn('[Observation truncated', text)
                self.assertEqual(audit['original_observation'], body)
                self.assertTrue(text.endswith('\n'))
                self.assertEqual(self.fit(observation=body), (text, audit))

    def test_actions_alone_over_budget_fail_with_specific_evidence(self):
        with self.assertRaises(WebshopInputBudgetError) as caught:
            self.fit(actions='click[' + 'option' * 2000 + ']')
        self.assertGreater(caught.exception.evidence['fixed_tokens'], 7000)
        self.assertIn('actions alone exceed cap', caught.exception.evidence['reason'])

    def test_product_pages_root_restart_mixed_slots_and_old_payload_are_not_reparsed_as_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = config_for(Path(tmp), 'webshop', 'full')
            config.omr.input_profile_id = INPUT_PROFILE
            manager = native_manager(config, [])
            manager.omr_tokenizer = CharacterTokenizer()
            root = 'WebShop [SEP] Instruction: [SEP] buy blue [SEP] Search'
            page = 'Product B0000 [SEP] blue [SEP] Buy Now'
            info = dict(goal_id='verified-goal', omr_instruction_text='buy blue',
                available_actions={'has_search_bar': False, 'clickables': ['blue', 'buy now']})
            root_packet = dict(observation=root, info=deepcopy(info), manager=None)
            initial, _ = render_starts(manager, 'webshop', [root_packet])
            manager.memory.store({'text_obs': [manager.pre_text_obs[0]], 'action': ['search[blue]']})
            manager.pre_text_obs = manager.format_obs([page])
            old_text = 'old profile rendered history'
            state = dict(task_text='buy blue', pre_text_obs=manager.pre_text_obs[0], history_length=2,
                         memory=deepcopy(manager.memory[0]), rendered_observation=old_text)
            packet = dict(observation=page, info=deepcopy(info), manager=state)
            before = deepcopy(packet)
            rendered, _ = render_starts(manager, 'webshop', [root_packet, packet])
            self.assertIn('Product B0000', rendered['text'][1])
            self.assertIn('search[blue]', rendered['text'][1])
            self.assertNotIn('search[blue]', rendered['text'][0])
            self.assertEqual([len(row) for row in manager.memory._data], [0, 1])
            self.assertEqual(rendered['input_audit'][1]['previous_input_profile_id'], 'webshop-input-v1')
            self.assertEqual(packet, before)
            broken = deepcopy(packet)
            broken['info']['omr_instruction_text'] = 'other task'
            with self.assertRaisesRegex(ValueError, 'instruction differs'):
                render_starts(manager, 'webshop', [broken])
            broken = deepcopy(packet)
            broken['manager']['pre_text_obs'] = 'other product'
            with self.assertRaisesRegex(ValueError, 'native page'):
                render_starts(manager, 'webshop', [broken])


if __name__ == '__main__':
    unittest.main()
