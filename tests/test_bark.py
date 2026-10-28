"""Optional Underdog Bark 3.0 catalog entry and its <unk> output."""

import threading
from unittest.mock import MagicMock

from vocalinux.speech_recognition.recognition_manager import SpeechRecognitionManager
from vocalinux.utils import parakeet_model_info as models
from vocalinux.utils.model_checksums import expected_for

MODEL = "bark-3.0"


def test_bark_is_optional_pinned_and_keeps_its_license_files() -> None:
    """bark-3.0 is offered but not recommended, and every file it downloads is pinned."""
    assert models.RECOMMENDED_MODEL == "v3-european"
    assert MODEL in models.MODEL_SIZES
    assert models.model_files(MODEL)[:4] == models.MODEL_FILES
    assert {"LICENSE", "NOTICE"} <= set(models.model_files(MODEL))
    for name in models.model_files(MODEL):
        assert expected_for(models.manifest_key(MODEL, name)) is not None
    assert models.get_model_file_url(MODEL, "encoder.int8.onnx") == (
        "https://huggingface.co/velddev/Underdog-Bark-3.0-sherpa-onnx/resolve/"
        "f438275bc431c58010eadb982dd0f8693ff4364f/encoder.int8.onnx?download=true"
    )


def test_unknown_word_token_is_dropped_from_the_transcript() -> None:
    """Parakeet transcripts drop <unk> and collapse the spaces it leaves."""
    manager = SpeechRecognitionManager.__new__(SpeechRecognitionManager)
    manager._model_lock = threading.Lock()
    manager.model = MagicMock()
    manager.model.create_stream.return_value.result.text = " like a shampooer,<unk>Next man.<unk>"
    assert manager._transcribe_with_parakeet([b"\x00\x00" * 160]) == "like a shampooer, Next man."
