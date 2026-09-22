"""Offline adapter contracts; never load weights or invoke a paid API."""

import json
from unittest.mock import Mock

import pytest

from ultrafast_laya import UltrafastLaya, normalize_question


def test_structured_instructions_are_lossless_json():
    instructions = {'goal': '选择 EUR', 'rules': ['Current state', 'Do not repeat']}
    q = normalize_question({'type': 'choice', 'criteria': {'2:1': {'value': 'EUR'}}, 'instructions': instructions})
    assert json.loads(q['instructions']) == instructions
    assert q['criteria'] == {'2:1': {'value': 'EUR'}}


@pytest.mark.parametrize('criteria', [{}, [], {str(i): '' for i in range(256)}, {'': 'empty'}, {1: 'number'}])
def test_invalid_choices_rejected(criteria):
    with pytest.raises(ValueError):
        normalize_question({'type': 'choice', 'criteria': criteria})


def test_singleton_does_not_run_model_or_invent_candidates():
    backend = UltrafastLaya.__new__(UltrafastLaya)
    backend.identity = {'variant': 'test'}
    backend.model = Mock(side_effect=AssertionError('No forward expected'))
    result = backend.evaluate({'state': {}, 'questions': {'select_target': {
        'type': 'choice', 'criteria': {'3:2': {'element': 'Currency EUR'}}, 'instructions': {'goal': 'EUR'}}}})
    assert result['answers']['select_target'] == {
        'type': 'choice', 'choice': '3:2', 'confidence': 1.0, 'probabilities': {'3:2': 1.0}}
    assert result['usage']['forward_passes'] == 0
    assert result['usage']['singleton_heads'] == ['select_target']
    backend.model.assert_not_called()
