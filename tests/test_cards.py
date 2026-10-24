"""
tests/test_cards.py – pytest suite for cards.py and app.generate_with_fallback.
No network calls. All Gemini API calls are mocked.
"""

import csv
import io
import json
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from cards import build_prompt, parse_cards, cards_to_csv, shrink_image

# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _make_response(text: str) -> MagicMock:
    """Return a mock that looks like a google-genai GenerateContentResponse."""
    r = MagicMock()
    r.text = text
    return r


def _api_error(msg: str) -> Exception:
    """Return a plain Exception whose str() contains *msg*."""
    return Exception(msg)


# =========================================================================== #
# parse_cards                                                                  #
# =========================================================================== #

class TestParseCards:
    # --- happy path --------------------------------------------------------- #

    def test_valid_json_single_card(self):
        raw = json.dumps([{"front": "What is F = ma?", "back": "Newton's 2nd law."}])
        cards = parse_cards(raw)
        assert len(cards) == 1
        assert cards[0]["front"] == "What is F = ma?"
        assert cards[0]["back"] == "Newton's 2nd law."

    def test_valid_json_multiple_cards(self):
        data = [{"front": f"Q{i}", "back": f"A{i}"} for i in range(5)]
        cards = parse_cards(json.dumps(data))
        assert len(cards) == 5
        for i, card in enumerate(cards):
            assert card["front"] == f"Q{i}"
            assert card["back"]  == f"A{i}"

    def test_strips_json_fenced_block(self):
        """```json\\n[...]\\n``` must be parsed correctly."""
        inner = json.dumps([{"front": "Q", "back": "A"}])
        raw = f"```json\n{inner}\n```"
        cards = parse_cards(raw)
        assert cards[0]["front"] == "Q"

    def test_strips_generic_fences(self):
        inner = json.dumps([{"front": "Q", "back": "A"}])
        raw = f"```\n{inner}\n```"
        cards = parse_cards(raw)
        assert len(cards) == 1

    def test_strips_fences_with_trailing_whitespace(self):
        inner = json.dumps([{"front": "Q", "back": "A"}])
        raw = f"```json\n{inner}\n```   "
        cards = parse_cards(raw)
        assert cards[0]["back"] == "A"

    def test_extra_keys_are_ignored(self):
        raw = json.dumps([{"front": "Q", "back": "A", "extra": "ignored"}])
        cards = parse_cards(raw)
        assert cards[0]["front"] == "Q"
        assert "extra" not in cards[0]

    def test_whitespace_trimmed(self):
        raw = json.dumps([{"front": "  Q  ", "back": "  A  "}])
        cards = parse_cards(raw)
        assert cards[0]["front"] == "Q"
        assert cards[0]["back"]  == "A"

    # --- invalid JSON ------------------------------------------------------- #

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_cards("this is not json at all")

    def test_truncated_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_cards('[{"front": "Q", "back":]')

    # --- wrong structure ---------------------------------------------------- #

    def test_non_list_json_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON list"):
            parse_cards(json.dumps({"front": "Q", "back": "A"}))

    def test_non_list_string_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON list"):
            parse_cards('"just a string"')

    def test_non_list_number_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON list"):
            parse_cards("42")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            parse_cards("[]")

    # --- missing keys ------------------------------------------------------- #

    def test_missing_front_raises(self):
        with pytest.raises(ValueError, match="front"):
            parse_cards(json.dumps([{"back": "A"}]))

    def test_missing_back_raises(self):
        with pytest.raises(ValueError, match="back"):
            parse_cards(json.dumps([{"front": "Q"}]))

    def test_missing_both_keys_raises(self):
        with pytest.raises(ValueError, match="front"):
            parse_cards(json.dumps([{"question": "Q", "answer": "A"}]))

    def test_non_dict_item_raises(self):
        with pytest.raises(ValueError, match="not an object"):
            parse_cards(json.dumps(["not a dict"]))

    # --- edge cases --------------------------------------------------------- #

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty response"):
            parse_cards("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty response"):
            parse_cards("   \n  ")


# =========================================================================== #
# cards_to_csv                                                                 #
# =========================================================================== #

class TestCardsToCSV:
    def _parse_csv(self, text: str) -> list[list[str]]:
        return list(csv.reader(io.StringIO(text)))

    def test_basic_csv(self):
        cards = [{"front": "Q1", "back": "A1"}, {"front": "Q2", "back": "A2"}]
        rows = self._parse_csv(cards_to_csv(cards))
        assert rows[0] == ["Q1", "A1"]
        assert rows[1] == ["Q2", "A2"]

    def test_no_header_row(self):
        cards = [{"front": "Q", "back": "A"}]
        rows = self._parse_csv(cards_to_csv(cards))
        assert rows[0] == ["Q", "A"]
        assert len(rows) == 1

    def test_comma_in_content(self):
        cards = [{"front": "What are SI units?", "back": "kg, m, s, A, K"}]
        rows = self._parse_csv(cards_to_csv(cards))
        assert rows[0][1] == "kg, m, s, A, K"

    def test_double_quote_in_content(self):
        cards = [{"front": 'He said "hello"', "back": 'Answer: "hi"'}]
        rows = self._parse_csv(cards_to_csv(cards))
        assert rows[0][0] == 'He said "hello"'
        assert rows[0][1] == 'Answer: "hi"'

    def test_newline_in_content(self):
        cards = [{"front": "Multi\nline front", "back": "Multi\nline back"}]
        rows = self._parse_csv(cards_to_csv(cards))
        assert rows[0][0] == "Multi\nline front"
        assert rows[0][1] == "Multi\nline back"

    def test_empty_cards_list(self):
        assert cards_to_csv([]) == ""

    def test_returns_string(self):
        assert isinstance(cards_to_csv([{"front": "Q", "back": "A"}]), str)


# =========================================================================== #
# build_prompt                                                                 #
# =========================================================================== #

class TestBuildPrompt:
    def test_contains_card_count(self):
        assert "10" in build_prompt(10, "English")

    def test_english_language(self):
        prompt = build_prompt(5, "English")
        assert "English" in prompt
        assert "Hindi" not in prompt

    def test_hindi_language(self):
        assert "Hindi" in build_prompt(5, "Hindi")

    def test_json_instruction_present(self):
        assert "JSON" in build_prompt(5, "English")

    def test_returns_string(self):
        assert isinstance(build_prompt(8, "English"), str)


# =========================================================================== #
# shrink_image                                                                 #
# =========================================================================== #

class TestShrinkImage:
    def _make_png(self, width: int, height: int) -> bytes:
        img = Image.new("RGB", (width, height), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_large_landscape_shrunk(self):
        raw = self._make_png(3200, 1800)
        out_bytes, mime = shrink_image(raw, max_side=1600)
        assert mime == "image/jpeg"
        img = Image.open(io.BytesIO(out_bytes))
        assert max(img.size) <= 1600

    def test_large_portrait_shrunk(self):
        raw = self._make_png(1000, 4000)
        out_bytes, mime = shrink_image(raw, max_side=1600)
        img = Image.open(io.BytesIO(out_bytes))
        assert max(img.size) <= 1600

    def test_small_image_not_enlarged(self):
        raw = self._make_png(400, 300)
        out_bytes, _ = shrink_image(raw, max_side=1600)
        img = Image.open(io.BytesIO(out_bytes))
        assert img.size == (400, 300)

    def test_exact_boundary_not_resized(self):
        raw = self._make_png(1600, 900)
        out_bytes, _ = shrink_image(raw, max_side=1600)
        img = Image.open(io.BytesIO(out_bytes))
        assert max(img.size) <= 1600

    def test_returns_jpeg_mime(self):
        raw = self._make_png(100, 100)
        _, mime = shrink_image(raw)
        assert mime == "image/jpeg"

    def test_output_is_jpeg(self):
        raw = self._make_png(100, 100)
        out_bytes, _ = shrink_image(raw)
        img = Image.open(io.BytesIO(out_bytes))
        assert img.format == "JPEG"

    def test_rgba_converted_to_rgb(self):
        """RGBA PNG must not cause JPEG save to crash."""
        img = Image.new("RGBA", (200, 200), color=(0, 128, 255, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out_bytes, _ = shrink_image(buf.getvalue())
        result = Image.open(io.BytesIO(out_bytes))
        assert result.mode == "RGB"


# =========================================================================== #
# generate_with_fallback (mocked – no network)                                #
# =========================================================================== #

# We patch 'app.genai.Client' so the real SDK is never called.

GOOD_JSON = json.dumps([{"front": "Q", "back": "A"}])


class TestGenerateWithFallback:
    """All tests mock genai.Client so no network calls are made."""

    def _patch_client(self, side_effects_per_model: dict):
        """
        Build a mock genai.Client whose generate_content raises or returns
        depending on which model is called.

        side_effects_per_model: {model_name: [effect, effect, ...]}
          Each effect is either an Exception instance (to raise) or a string
          (to return as response.text).
        """
        effects_copy = {k: list(v) for k, v in side_effects_per_model.items()}

        def fake_generate(model, contents, **kwargs):
            effects = effects_copy.get(model, [])
            if not effects:
                raise Exception(f"404 model not found: {model}")
            effect = effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            resp = MagicMock()
            resp.text = effect
            return resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate
        return mock_client

    # ----------------------------------------------------------------------- #
    # Happy paths                                                              #
    # ----------------------------------------------------------------------- #

    def test_success_on_first_model(self):
        from app import generate_with_fallback, MODELS

        client = self._patch_client({MODELS[0]: [GOOD_JSON]})
        with patch("app.genai.Client", return_value=client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            text, model = generate_with_fallback(["dummy_contents"])

        assert text == GOOD_JSON
        assert model == MODELS[0]

    def test_preferred_model_tried_first(self):
        from app import generate_with_fallback, MODELS

        client = self._patch_client({MODELS[1]: [GOOD_JSON], MODELS[0]: []})
        with patch("app.genai.Client", return_value=client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            text, model = generate_with_fallback(
                ["dummy_contents"], preferred_model=MODELS[1]
            )

        assert model == MODELS[1]

    # ----------------------------------------------------------------------- #
    # Retry on 500/503                                                         #
    # ----------------------------------------------------------------------- #

    def test_500_retried_then_success(self):
        """First attempt gets 500, second attempt succeeds on the same model."""
        from app import generate_with_fallback, MODELS

        client = self._patch_client({
            MODELS[0]: [_api_error("500 INTERNAL"), GOOD_JSON],
        })
        with patch("app.genai.Client", return_value=client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.time.sleep"):          # skip actual sleep
            text, model = generate_with_fallback(["dummy"])

        assert model == MODELS[0]
        assert text  == GOOD_JSON

    def test_503_retried_twice_then_fallback(self):
        """Model 0 fails all retries → falls back to model 1 which succeeds."""
        from app import generate_with_fallback, MODELS

        client = self._patch_client({
            MODELS[0]: [
                _api_error("503 UNAVAILABLE"),
                _api_error("503 UNAVAILABLE"),
                _api_error("503 UNAVAILABLE"),
            ],
            MODELS[1]: [GOOD_JSON],
        })
        with patch("app.genai.Client", return_value=client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.time.sleep"):
            text, model = generate_with_fallback(["dummy"])

        assert model == MODELS[1]
        assert text  == GOOD_JSON

    # ----------------------------------------------------------------------- #
    # Model fallback                                                           #
    # ----------------------------------------------------------------------- #

    def test_404_skips_to_next_model_immediately(self):
        """404 on model 0 must not trigger retries – go straight to model 1."""
        from app import generate_with_fallback, MODELS

        call_counts: dict[str, int] = {MODELS[0]: 0, MODELS[1]: 0}

        def fake_generate(model, contents, **kwargs):
            call_counts[model] = call_counts.get(model, 0) + 1
            if model == MODELS[0]:
                raise _api_error("404 model not_found")
            resp = MagicMock()
            resp.text = GOOD_JSON
            return resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate

        with patch("app.genai.Client", return_value=mock_client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.time.sleep"):
            text, model = generate_with_fallback(["dummy"])

        # Model 0 must have been called exactly once (no retries on 404)
        assert call_counts[MODELS[0]] == 1
        assert model == MODELS[1]

    def test_all_models_fail_raises_runtime_error(self):
        """When every model exhausts all attempts, a readable RuntimeError is raised."""
        from app import generate_with_fallback, MODELS

        client = self._patch_client({
            m: [_api_error("500 INTERNAL")] * 3 for m in MODELS
        })
        with patch("app.genai.Client", return_value=client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.time.sleep"):
            with pytest.raises(RuntimeError, match="All models failed"):
                generate_with_fallback(["dummy"])

    # ----------------------------------------------------------------------- #
    # Fatal / auth errors                                                      #
    # ----------------------------------------------------------------------- #

    def test_invalid_key_stops_immediately(self):
        """A 403 / bad API key must raise RuntimeError without trying other models."""
        from app import generate_with_fallback, MODELS

        call_count = {"n": 0}

        def fake_generate(model, contents, **kwargs):
            call_count["n"] += 1
            raise _api_error("403 permission denied invalid key")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate

        with patch("app.genai.Client", return_value=mock_client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with pytest.raises(RuntimeError, match="Authentication"):
                generate_with_fallback(["dummy"])

        # Should have stopped after the very first call
        assert call_count["n"] == 1

    def test_missing_api_key_raises_immediately(self):
        """No key → RuntimeError before any API call is made."""
        from app import generate_with_fallback

        mock_client = MagicMock()
        with patch("app.genai.Client", return_value=mock_client), \
             patch.dict("os.environ", {}, clear=True):
            # Remove GEMINI_API_KEY if it happens to be set in the test environment
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                generate_with_fallback(["dummy"])

        mock_client.models.generate_content.assert_not_called()

    # ----------------------------------------------------------------------- #
    # Full flow round-trip                                                     #
    # ----------------------------------------------------------------------- #

    def test_full_flow_parse_and_csv(self):
        """generate_with_fallback → parse_cards → cards_to_csv round-trip."""
        from app import generate_with_fallback, MODELS

        client = self._patch_client({MODELS[0]: [GOOD_JSON]})
        with patch("app.genai.Client", return_value=client), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            raw, model = generate_with_fallback(["dummy"])

        cards = parse_cards(raw)
        csv_text = cards_to_csv(cards)
        rows = list(csv.reader(io.StringIO(csv_text)))
        assert rows[0] == ["Q", "A"]
