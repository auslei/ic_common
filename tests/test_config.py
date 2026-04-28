import os
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.common.config import (
    AppConfig, 
    load_config_from_path, 
    find_config, 
    load_config,
    reload_config
)

def test_app_config_defaults():
    cfg = AppConfig()
    assert cfg.api_key == "dev-secret-key"
    assert cfg.core_app.company_name == "InvestorIQ"
    assert cfg.llm.provider == "ollama"

def test_load_config_from_path_missing(tmp_path):
    missing_path = tmp_path / "nonexistent.yaml"
    cfg = load_config_from_path(missing_path)
    assert isinstance(cfg, AppConfig)
    # Should return defaults if file missing

def test_load_config_from_path_valid(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    data = {
        "api_key": "prod-key",
        "core_app": {
            "company_name": "CustomCorp"
        }
    }
    with open(cfg_file, "w") as f:
        yaml.dump(data, f)
    
    cfg = load_config_from_path(cfg_file)
    assert cfg.api_key == "prod-key"
    assert cfg.core_app.company_name == "CustomCorp"

def test_load_config_from_path_invalid(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    with open(cfg_file, "w") as f:
        f.write("invalid: [yaml: content")
    
    with pytest.raises(Exception):
        load_config_from_path(cfg_file)

def test_find_config_env_var(tmp_path):
    custom_path = tmp_path / "custom_config.yaml"
    custom_path.touch()
    
    with patch.dict(os.environ, {"PACKAGE_CONFIG_PATH": str(custom_path)}):
        assert find_config() == custom_path

def test_find_config_package_name(tmp_path, monkeypatch):
    # Change CWD to tmp_path for easier path testing
    monkeypatch.chdir(tmp_path)
    
    pkg_dir = tmp_path / "packages" / "my_pkg"
    pkg_dir.mkdir(parents=True)
    pkg_cfg = pkg_dir / "config.yaml"
    pkg_cfg.touch()
    
    with patch.dict(os.environ, {"PACKAGE_NAME": "my_pkg"}):
        # Should find packages/my_pkg/config.yaml
        assert find_config() == Path("packages/my_pkg/config.yaml")

def test_find_config_src_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    src_dir = tmp_path / "src" / "my_pkg"
    src_dir.mkdir(parents=True)
    src_cfg = src_dir / "config.yaml"
    src_cfg.touch()
    
    with patch.dict(os.environ, {"PACKAGE_NAME": "my_pkg"}):
        # Should find src/my_pkg/config.yaml
        assert find_config() == Path("src/my_pkg/config.yaml")

def test_find_config_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {"PACKAGE_NAME": "missing_pkg"}):
        assert find_config() == Path("config.yaml")

def test_load_and_reload_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump({"api_key": "first"}, f)
    
    cfg = load_config()
    assert cfg.api_key == "first"
    
    # Update file
    with open(cfg_file, "w") as f:
        yaml.dump({"api_key": "second"}, f)
    
    # Reload
    from src.common import config as config_mod
    config_mod.reload_config()
    assert config_mod.config.api_key == "second"
