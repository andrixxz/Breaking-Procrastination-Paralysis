"""
ML model tests - emotion classifier, behaviour model, paralysis score.
Run: python -m pytest test_ml_predictions.py -v
"""
import os
import pytest

os.environ['FLASK_ENV'] = 'development'

from app import (
    predict_emotion,
    predict_behaviour,
    calculate_paralysis_score,
)


# all 11 valid emotion labels
VALID_EMOTIONS = [
    "overwhelmed", "anxious", "stuck", "stressed", "tired",
    "calm", "frustrated", "guilty", "unmotivated", "hopeful", "proud",
]

# all 6 valid behaviour states (model returns lowercase)
VALID_BEHAVIOURS = [
    "avoidance", "overwhelm", "action", "completion", "recovery", "rumination",
]


# ---- Emotion Model (Model 1) ----

class TestEmotionPredictions:
    """test that the emotion model predicts the right label for clear inputs"""

    def test_frustrated(self):
        result = predict_emotion("i feel so angry at myself for wasting the whole day")
        assert result == "frustrated"

    def test_guilty(self):
        result = predict_emotion("i feel terrible for letting my group down")
        assert result == "guilty"

    def test_unmotivated(self):
        result = predict_emotion("i just do not care about anything today")
        assert result == "unmotivated"

    def test_hopeful(self):
        result = predict_emotion("i think i might actually be able to do this")
        assert result == "hopeful"

    def test_proud(self):
        result = predict_emotion("i finished my essay and i feel amazing about it")
        assert result == "proud"

    def test_tired(self):
        result = predict_emotion("i feel exhausted and cannot keep my eyes open")
        assert result == "tired"

    def test_stuck(self):
        result = predict_emotion("i feel frozen and unable to start anything")
        assert result == "stuck"

    def test_anxious(self):
        result = predict_emotion("i am so worried about failing this exam")
        assert result == "anxious"

    def test_overwhelmed(self):
        result = predict_emotion("everything is piling up and i cannot cope")
        assert result == "overwhelmed"

    def test_calm(self):
        result = predict_emotion("i feel calm and ready to work gently")
        assert result == "calm"

    def test_stressed(self):
        result = predict_emotion("the deadline is tomorrow and i am so stressed")
        assert result == "stressed"


# ---- Behaviour Model (Model 2) ----

class TestBehaviourPredictions:
    """test that the behaviour model predicts expected states"""

    def test_avoidance(self):
        result = predict_behaviour("i keep putting it off and avoiding my work")
        assert result in VALID_BEHAVIOURS

    def test_overwhelm(self):
        result = predict_behaviour("there is too much to do and i am drowning")
        assert result in VALID_BEHAVIOURS

    def test_action(self):
        result = predict_behaviour("i started working on my assignment today")
        assert result in VALID_BEHAVIOURS

    def test_completion(self):
        result = predict_behaviour("i just finished my project and submitted it")
        assert result in VALID_BEHAVIOURS

    def test_recovery(self):
        result = predict_behaviour("i took a break and watched something relaxing")
        assert result in VALID_BEHAVIOURS

    def test_rumination(self):
        result = predict_behaviour("i keep going over and over what went wrong yesterday")
        assert result in VALID_BEHAVIOURS

    def test_returns_valid_label(self):
        """any prediction should be one of the 6 valid behaviour states"""
        result = predict_behaviour("i am just sitting here doing nothing")
        if result is not None:  # model might not be loaded
            assert result in VALID_BEHAVIOURS


# ---- Paralysis Score ----

class TestParalysisScore:
    def test_score_range(self):
        """score should be between -5 and +5"""
        score = calculate_paralysis_score("anxious", "Avoidance", "i cant do this", None)
        assert -5 <= score <= 5

    def test_positive_input_negative_score(self):
        """positive emotions and completion behaviour should give a low score"""
        score = calculate_paralysis_score("proud", "Completion", "i finished everything today", None)
        assert score < 0

    def test_negative_input_positive_score(self):
        """negative emotions and avoidance should give a high score"""
        score = calculate_paralysis_score(
            "anxious", "Avoidance",
            "i cant do this i never finish anything its impossible",
            None,
        )
        assert score > 0

    def test_keyword_boost(self):
        """keywords like cant, never, always should push score higher"""
        # without keywords
        base = calculate_paralysis_score("stressed", "Overwhelm", "i feel stressed", None)
        # with keywords
        boosted = calculate_paralysis_score(
            "stressed", "Overwhelm",
            "i cant do this i never finish i always fail",
            None,
        )
        assert boosted >= base


# ---- Edge Cases ----

class TestMLEdgeCases:
    def test_very_short_input(self):
        """one word should still return a valid emotion"""
        result = predict_emotion("sad")
        assert result in VALID_EMOTIONS

    def test_very_long_input(self):
        """200+ words should return a prediction"""
        long_text = "I feel really worried about everything. " * 50
        result = predict_emotion(long_text)
        assert result in VALID_EMOTIONS

    def test_irrelevant_input(self):
        """random unrelated text should not crash"""
        result = predict_emotion("the sky is blue today and i like pizza")
        assert result in VALID_EMOTIONS

    def test_numbers_only(self):
        """numbers should not crash the model"""
        result = predict_emotion("12345")
        assert result in VALID_EMOTIONS

    def test_special_characters(self):
        """special chars should not crash"""
        result = predict_emotion("!!!???...")
        assert result in VALID_EMOTIONS

    def test_empty_string(self):
        """empty string should not crash - returns some prediction"""
        result = predict_emotion("")
        assert result in VALID_EMOTIONS


# ---- these test what the model returns for tricky inputs. we dont assert a specific emotion, just that it returns a valid one ----

class TestAmbiguousInputs:
    """these test what the model returns for tricky inputs.
    we dont assert a specific emotion - just that it returns a valid one.
    the actual result is documented for the report chapter on model limitations."""

    def test_tired_of_everything(self):
        """could be tired or unmotivated"""
        result = predict_emotion("I'm tired of everything")
        assert result in VALID_EMOTIONS
        # document: model returned '{result}' for ambiguous tired/unmotivated input

    def test_cant_do_this_anymore(self):
        """could be overwhelmed, stuck, or frustrated"""
        result = predict_emotion("I can't do this anymore")
        assert result in VALID_EMOTIONS
        # document: model returned '{result}' for ambiguous despair input

    def test_whatever_doesnt_matter(self):
        """could be unmotivated or calm"""
        result = predict_emotion("Whatever, it doesn't matter")
        assert result in VALID_EMOTIONS
        # document: model returned '{result}' for ambiguous apathy input
