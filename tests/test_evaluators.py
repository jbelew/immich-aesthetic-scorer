"""Unit tests for the unified image evaluation engine and model adapters."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from immich_aesthetic_scorer.evaluators import (  # noqa: E402
    EvaluationResult,
    FakeEvaluator,
    PyIQAEvaluator,
)


class TestFakeEvaluator:
    def test_evaluates_with_default_score(self) -> None:
        evaluator = FakeEvaluator(default_score=7.5, model_id="fake-model")
        result = evaluator.evaluate(b"img-bytes-1")

        assert isinstance(result, EvaluationResult)
        assert result.raw_score == 7.5
        assert result.score == 75
        assert result.reason == "Fake score: 7.50"
        assert evaluator.model_id == "fake-model"
        assert evaluator.required_max_dimension == 512
        assert evaluator.evaluated_images == [b"img-bytes-1"]

    def test_evaluates_with_score_map_and_callable(self) -> None:
        score_map = {
            b"good": 9.2,
            b"custom": EvaluationResult(raw_score=88.0, reason="Custom reason", score=88),
        }
        evaluator = FakeEvaluator(
            scores=score_map,
            default_score=5.0,
            required_max_dimension=None,
        )

        res_good = evaluator.evaluate(b"good")
        assert res_good.raw_score == 9.2

        res_custom = evaluator.evaluate(b"custom")
        assert res_custom.raw_score == 88.0
        assert res_custom.reason == "Custom reason"
        assert res_custom.score == 88

        res_fallback = evaluator.evaluate(b"unknown")
        assert res_fallback.raw_score == 5.0
        assert evaluator.required_max_dimension is None

        fn_evaluator = FakeEvaluator(scores=lambda data: float(len(data)))
        assert fn_evaluator.evaluate(b"abcd").raw_score == 4.0


class TestLocalClipEvaluator:
    def test_negotiates_cuda_capability_and_falls_back_to_cpu(self) -> None:
        from unittest.mock import MagicMock

        from immich_aesthetic_scorer.evaluators import negotiate_torch_device

        mock_torch_cuda_7 = MagicMock()
        mock_torch_cuda_7.cuda.is_available.return_value = True
        mock_torch_cuda_7.cuda.get_device_capability.return_value = (8, 6)
        assert negotiate_torch_device(mock_torch_cuda_7) == "cuda"

        mock_torch_cuda_6 = MagicMock()
        mock_torch_cuda_6.cuda.is_available.return_value = True
        mock_torch_cuda_6.cuda.get_device_capability.return_value = (6, 1)
        assert negotiate_torch_device(mock_torch_cuda_6) == "cpu"

        mock_torch_err = MagicMock()
        mock_torch_err.cuda.is_available.return_value = True
        mock_torch_err.cuda.get_device_capability.side_effect = RuntimeError("CUDA driver error")
        assert negotiate_torch_device(mock_torch_err) == "cpu"

        mock_torch_no_cuda = MagicMock()
        mock_torch_no_cuda.cuda.is_available.return_value = False
        assert negotiate_torch_device(mock_torch_no_cuda) == "cpu"

    def test_evaluates_rsinema_backbone_under_no_grad(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io
        from unittest.mock import MagicMock

        from PIL import Image

        import immich_aesthetic_scorer.evaluators as eval_mod
        from immich_aesthetic_scorer.evaluators import LocalClipEvaluator

        img = Image.new("RGB", (64, 64), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (7, 5)

        mock_output_item = MagicMock()
        mock_output_item.item.return_value = 4.25
        mock_output_row = [mock_output_item]
        mock_outputs = MagicMock()
        mock_outputs.shape = (1, 7)
        mock_outputs.__getitem__.side_effect = lambda idx: mock_output_row

        mock_model = MagicMock(return_value=mock_outputs)
        mock_scorer_cls = MagicMock(return_value=mock_model)

        mock_pixel_values = MagicMock()
        mock_pixel_values.to.return_value = "device_pixels"
        mock_processor = MagicMock(return_value={"pixel_values": mock_pixel_values})
        mock_processor_cls = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        monkeypatch.setattr(eval_mod, "torch", mock_torch)
        monkeypatch.setattr(eval_mod, "CLIPProcessor", mock_processor_cls)
        monkeypatch.setattr(eval_mod, "RsinemaAestheticScorer", mock_scorer_cls)

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = "/fake/model.pt"
        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", mock_hf)

        evaluator = LocalClipEvaluator(model_id="rsinema/aesthetic-scorer")
        assert evaluator.required_max_dimension == 512

        res = evaluator.evaluate(img_bytes)
        assert res.raw_score == 8.5  # 4.25 * 2.0
        assert res.score == 85
        assert res.reason is None
        mock_model.to.assert_called_once_with("cuda")
        mock_torch.no_grad.assert_called_once()

    def test_evaluates_siglip_backbone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io
        import sys
        from unittest.mock import MagicMock

        from PIL import Image

        import immich_aesthetic_scorer.evaluators as eval_mod
        from immich_aesthetic_scorer.evaluators import LocalClipEvaluator

        img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setattr(eval_mod, "torch", mock_torch)
        monkeypatch.setattr(eval_mod, "CLIPProcessor", MagicMock())

        mock_hf = MagicMock()
        mock_hf.snapshot_download.return_value = "/fake/siglip_dir"
        monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hf)

        mock_siglip_instance = MagicMock()
        mock_siglip_instance.rate.return_value = 7.84
        mock_predict_mod = MagicMock()
        mock_predict_mod.AestheticScorer.from_local.return_value = mock_siglip_instance
        monkeypatch.setitem(sys.modules, "predict", mock_predict_mod)

        evaluator = LocalClipEvaluator(model_id="somepago/AestheticSigLIP")
        res = evaluator.evaluate(img_bytes)

        assert res.raw_score == 7.84
        assert res.score == 78
        assert res.reason is None
        mock_torch.no_grad.assert_called_once()

    def test_evaluates_aesthetics_predictor_v2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io
        import sys
        from unittest.mock import MagicMock

        from PIL import Image

        import immich_aesthetic_scorer.evaluators as eval_mod
        from immich_aesthetic_scorer.evaluators import LocalClipEvaluator

        img = Image.new("RGB", (32, 32), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setattr(eval_mod, "torch", mock_torch)

        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_processor = MagicMock(return_value={"pixel_values": mock_tensor})
        mock_processor_cls = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        monkeypatch.setattr(eval_mod, "CLIPProcessor", mock_processor_cls)

        mock_outputs = MagicMock()
        mock_outputs.logits.item.return_value = 6.4
        mock_model = MagicMock(return_value=mock_outputs)
        mock_model.to.return_value = mock_model

        mock_ap_mod = MagicMock()
        mock_ap_mod.AestheticsPredictorV2Linear.from_pretrained.return_value = mock_model
        monkeypatch.setitem(sys.modules, "aesthetics_predictor", mock_ap_mod)

        evaluator = LocalClipEvaluator(
            model_id="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        )
        res = evaluator.evaluate(img_bytes)

        assert res.raw_score == 6.4
        assert res.score == 64
        assert res.reason is None
        mock_torch.no_grad.assert_called_once()


class TestPyIQAEvaluator:
    def test_evaluates_in_memory_without_leaking_temp_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike[str]
    ) -> None:
        import io
        import tempfile
        from unittest.mock import MagicMock

        from PIL import Image

        import immich_aesthetic_scorer.evaluators as eval_mod

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

        img = Image.new("RGB", (48, 48), color="yellow")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)

        mock_score_tensor = MagicMock()
        mock_score_tensor.item.return_value = 67.89
        mock_metric = MagicMock(return_value=mock_score_tensor)

        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_metric

        monkeypatch.setattr(eval_mod, "torch", mock_torch)
        monkeypatch.setattr(eval_mod, "pyiqa", mock_pyiqa)

        evaluator = PyIQAEvaluator(model_id="musiq-spaq")
        assert evaluator.model_id == "musiq-spaq"
        assert evaluator.required_max_dimension is None

        res = evaluator.evaluate(img_bytes)
        assert res.raw_score == 67.89
        assert res.reason is None
        mock_pyiqa.create_metric.assert_called_once_with("musiq-spaq", device="cuda")
        mock_torch.no_grad.assert_called_once()

        # Verify PIL Image was passed directly in memory and no temp files exist
        called_arg = mock_metric.call_args[0][0]
        assert isinstance(called_arg, Image.Image)
        assert list(os.listdir(tmp_path)) == []

    def test_does_not_leak_temp_files_when_metric_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike[str]
    ) -> None:
        import io
        import tempfile
        from unittest.mock import MagicMock

        from PIL import Image

        import immich_aesthetic_scorer.evaluators as eval_mod

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

        img = Image.new("RGB", (32, 32), color="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_metric = MagicMock(side_effect=RuntimeError("metric failure"))
        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.return_value = mock_metric

        monkeypatch.setattr(eval_mod, "torch", mock_torch)
        monkeypatch.setattr(eval_mod, "pyiqa", mock_pyiqa)

        evaluator = PyIQAEvaluator(model_id="musiq-spaq")
        with pytest.raises(RuntimeError, match="metric failure"):
            evaluator.evaluate(img_bytes)

        assert list(os.listdir(tmp_path)) == []


class TestRateLimiter:
    def test_paces_consecutive_and_concurrent_calls(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from immich_aesthetic_scorer.evaluators import RateLimiter

        current_time = 100.0
        recorded_sleeps: list[float] = []
        lock = __import__("threading").Lock()

        def fake_clock() -> float:
            return current_time

        def fake_sleep(seconds: float) -> None:
            with lock:
                recorded_sleeps.append(seconds)

        limiter = RateLimiter(delay=4.0, clock=fake_clock, sleep=fake_sleep)

        # First call should not sleep; subsequent calls at the same timestamp reserve 4s and 8s slots
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: limiter.wait(), range(3)))

        assert sorted(recorded_sleeps) == [4.0, 8.0]


class TestRemoteVisionEvaluators:
    def test_gemini_evaluator_formats_payload_and_parses_json(self) -> None:
        import base64
        import json
        from unittest.mock import MagicMock

        from immich_aesthetic_scorer.evaluators import AESTHETIC_PROMPT, GeminiEvaluator

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {"score": 91, "reason": "Crisp subject and warm light."}
                                )
                            }
                        ]
                    }
                }
            ]
        }
        mock_session.post.return_value = mock_resp

        evaluator = GeminiEvaluator(
            api_key="test-gemini-key",
            model_id="gemini-2.5-flash",
            session=mock_session,
        )
        assert evaluator.model_id == "gemini-2.5-flash"
        assert evaluator.required_max_dimension == 512

        res = evaluator.evaluate(b"jpeg-bytes")
        assert res.raw_score == 91.0
        assert res.score == 91
        assert res.reason == "Crisp subject and warm light."

        mock_session.post.assert_called_once()
        call_url = mock_session.post.call_args[0][0]
        call_kwargs = mock_session.post.call_args[1]
        assert "gemini-2.5-flash:generateContent?key=test-gemini-key" in call_url
        parts = call_kwargs["json"]["contents"][0]["parts"]
        assert parts[0]["text"] == AESTHETIC_PROMPT
        assert parts[1]["inlineData"]["data"] == base64.b64encode(b"jpeg-bytes").decode("utf-8")

    def test_gemini_evaluator_retries_with_exponential_backoff_on_429_and_500(self) -> None:
        import json
        from unittest.mock import MagicMock

        from immich_aesthetic_scorer.evaluators import GeminiEvaluator

        resp_429 = MagicMock(status_code=429, text="Quota exceeded")
        resp_503 = MagicMock(status_code=503, text="Service unavailable")
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps({"score": 82, "reason": "Recovered"})}]}}
            ]
        }

        mock_session = MagicMock()
        mock_session.post.side_effect = [resp_429, resp_503, resp_200]
        sleeps: list[float] = []

        evaluator = GeminiEvaluator(
            api_key="key",
            initial_backoff=2.0,
            max_retries=4,
            session=mock_session,
            sleep=sleeps.append,
        )
        res = evaluator.evaluate(b"img")

        assert res.raw_score == 82.0
        assert res.reason == "Recovered"
        assert sleeps == [2.0, 4.0]
        assert mock_session.post.call_count == 3

    def test_gemini_evaluator_fails_immediately_on_non_retryable_4xx(self) -> None:
        from unittest.mock import MagicMock

        from immich_aesthetic_scorer.evaluators import GeminiEvaluator

        resp_400 = MagicMock(status_code=400, text="Bad Request")
        mock_session = MagicMock()
        mock_session.post.return_value = resp_400
        sleeps: list[float] = []

        evaluator = GeminiEvaluator(
            api_key="key",
            session=mock_session,
            sleep=sleeps.append,
        )
        with pytest.raises(Exception, match="Gemini API error 400"):
            evaluator.evaluate(b"img")

        assert sleeps == []
        assert mock_session.post.call_count == 1

    def test_openai_evaluator_formats_payload_and_retries_429(self) -> None:
        import base64
        import json
        from unittest.mock import MagicMock

        from immich_aesthetic_scorer.evaluators import OpenAIEvaluator

        resp_429 = MagicMock(status_code=429, text="Rate limit")
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 77, "reason": "Good framing"})}}
            ]
        }

        mock_session = MagicMock()
        mock_session.post.side_effect = [resp_429, resp_200]
        sleeps: list[float] = []

        evaluator = OpenAIEvaluator(
            api_key="sk-test",
            base_url="https://api.openai.com/v1/",
            model_id="gpt-4o-mini",
            initial_backoff=3.0,
            session=mock_session,
            sleep=sleeps.append,
        )
        assert evaluator.model_id == "gpt-4o-mini"
        assert evaluator.required_max_dimension == 512

        res = evaluator.evaluate(b"raw-img")
        assert res.raw_score == 77.0
        assert res.score == 77
        assert res.reason == "Good framing"
        assert sleeps == [3.0]

        call_url = mock_session.post.call_args[0][0]
        call_kwargs = mock_session.post.call_args[1]
        assert call_url == "https://api.openai.com/v1/chat/completions"
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test"
        expected_b64 = base64.b64encode(b"raw-img").decode("utf-8")
        content = call_kwargs["json"]["messages"][0]["content"]
        assert content[1]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


class TestCreateEvaluatorFactory:
    def test_dispatches_stage1_evaluators(self) -> None:
        from immich_aesthetic_scorer.evaluators import (
            FakeEvaluator,
            GeminiEvaluator,
            ImageEvaluator,
            LocalClipEvaluator,
            OpenAIEvaluator,
            create_evaluator,
        )

        local_eval = create_evaluator(
            scorer_type="local", local_model_id="rsinema/aesthetic-scorer"
        )
        assert isinstance(local_eval, LocalClipEvaluator)
        assert isinstance(local_eval, ImageEvaluator)
        assert local_eval.model_id == "rsinema/aesthetic-scorer"
        assert local_eval.required_max_dimension == 512

        gemini_eval = create_evaluator(
            scorer_type="gemini",
            gemini_key="g-key",
            gemini_model="gemini-2.5-flash",
        )
        assert isinstance(gemini_eval, GeminiEvaluator)
        assert gemini_eval.model_id == "gemini-2.5-flash"
        assert gemini_eval.required_max_dimension == 512

        openai_eval = create_evaluator(
            scorer_type="openai",
            openai_key="o-key",
            openai_model="gpt-4o",
        )
        assert isinstance(openai_eval, OpenAIEvaluator)
        assert openai_eval.model_id == "gpt-4o"
        assert openai_eval.required_max_dimension == 512

        fake_eval = create_evaluator(scorer_type="fake", model_id="fake-s1")
        assert isinstance(fake_eval, FakeEvaluator)
        assert fake_eval.model_id == "fake-s1"
        assert fake_eval.required_max_dimension == 512

    def test_dispatches_stage2_evaluators_with_appropriate_dimensions(self) -> None:
        from immich_aesthetic_scorer.evaluators import (
            GeminiEvaluator,
            OpenAIEvaluator,
            PyIQAEvaluator,
            create_evaluator,
        )

        pyiqa_eval = create_evaluator(stage=2, stage2_model="musiq-spaq")
        assert isinstance(pyiqa_eval, PyIQAEvaluator)
        assert pyiqa_eval.model_id == "musiq-spaq"
        assert pyiqa_eval.required_max_dimension is None

        gemini_s2 = create_evaluator(
            stage=2,
            stage2_model="gemini",
            gemini_key="g-key",
            gemini_model="gemini-2.5-pro",
        )
        assert isinstance(gemini_s2, GeminiEvaluator)
        assert gemini_s2.model_id == "gemini-2.5-pro"
        assert gemini_s2.required_max_dimension == 512

        openai_s2 = create_evaluator(
            stage=2,
            stage2_model="gpt-4o-mini",
            openai_key="o-key",
        )
        assert isinstance(openai_s2, OpenAIEvaluator)
        assert openai_s2.model_id == "gpt-4o-mini"
        assert openai_s2.required_max_dimension == 512

    def test_raises_value_error_for_missing_credentials_or_unknown_scorer(self) -> None:
        from immich_aesthetic_scorer.evaluators import create_evaluator

        with pytest.raises(ValueError, match="Gemini API Key"):
            create_evaluator(scorer_type="gemini", gemini_key=None)

        with pytest.raises(ValueError, match="OpenAI API Key"):
            create_evaluator(stage=2, stage2_model="gpt-4o", openai_key=None)

        with pytest.raises(ValueError, match="Unsupported scorer_type"):
            create_evaluator(scorer_type="unknown-provider")
