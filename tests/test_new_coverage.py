import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.common.LLMProvider import LLMProvider, LLMSettings
from src.common.file_utils import ensure_dir
from src.common.ml_utils import is_running_in_container, millisec_to_time, get_device, get_gpu_usage
from src.common.document_utils import _fix_markdown_lists_and_tables

def test_llm_provider():
    p = LLMProvider(provider="openai", model_name="gpt-4")
    assert p.provider == "openai"
    assert p.model_name == "gpt-4"
    assert p["provider"] == "openai"

def test_llm_settings():
    s = LLMSettings(provider="ollama", model_name="llama3", temperature=0.7, context_window=4096)
    assert s.temperature == 0.7
    assert s.context_window == 4096
    assert s["model_name"] == "llama3"

def test_ensure_dir(tmp_path):
    new_dir = tmp_path / "subdir" / "test"
    assert not new_dir.exists()
    p = ensure_dir(new_dir)
    assert new_dir.exists()
    assert isinstance(p, Path)

def test_is_running_in_container():
    # We can't easily test the True case without a container, 
    # but we can test that it runs without error.
    res = is_running_in_container()
    assert isinstance(res, bool)

def test_millisec_to_time():
    assert millisec_to_time(0) == "00:00:00"
    assert millisec_to_time(1000) == "00:00:01"
    assert millisec_to_time(3661000) == "01:01:01"

def test_fix_markdown_lists_and_tables():
    content = "Para 1\n- Item 1\n- Item 2"
    fixed = _fix_markdown_lists_and_tables(content)
    assert "Para 1\n\n- Item 1" in fixed
    
    content = "Table:\n| A | B |\n|---|---|"
    fixed = _fix_markdown_lists_and_tables(content)
    assert "Table:\n\n| A | B |" in fixed
    
    # Already fixed
    content = "Para 1\n\n* Item 1"
    assert _fix_markdown_lists_and_tables(content) == content

@patch("src.common.ml_utils.os.getenv")
def test_get_device_force(mock_getenv):
    mock_getenv.return_value = "cuda"
    assert get_device() == "cuda"
    mock_getenv.assert_called_with("FORCE_DEVICE")

@patch("src.common.ml_utils.os.path.exists")
def test_is_running_in_container_mock(mock_exists):
    mock_exists.side_effect = lambda p: p == "/.dockerenv"
    assert is_running_in_container() is True

def test_get_gpu_usage_mock():
    with patch("src.common.ml_utils.os.getenv", return_value="cpu"):
        assert get_gpu_usage() == 0.0
