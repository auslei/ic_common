import pytest
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from src.common.utils import (
    sanitize_project_name,
    render_markdown_to_html,
    normalize_finance_value,
    check_collection_exists,
    get_indexed_paths,
    check_project_processing_status
)

def test_sanitize_project_name():
    assert sanitize_project_name("My Project") == "My_Project"
    assert sanitize_project_name("Project!@#$%^&*()") == "Project"
    assert sanitize_project_name("Project-123_abc") == "Project-123_abc"
    assert sanitize_project_name("项目名称") == "项目名称"
    assert sanitize_project_name("  Trim  And  Space  ") == "_Trim_And_Space_"

def test_render_markdown_to_html():
    md = "# Hello\nThis is a test."
    html = render_markdown_to_html(md, title="Test Title")
    assert "<title>Test Title</title>" in html
    assert '<h1 id="hello">Hello</h1>' in html
    assert "<p>This is a test.</p>" in html
    assert "markdown-body" in html

@pytest.mark.parametrize("raw, target, expected", [
    ("1.2亿", "million", 120.0),
    ("3500万", "亿", 0.35),
    ("$50M", "1", 50000000.0),
    ("2000", "thousand", 2.0),
    ("￥100k", "万", 10.0),
    (None, "million", None),
    ("null", "million", None),
    ("invalid", "million", None),
    ("100", "invalid_unit", 0.0001), # Defaults to million (100 / 1,000,000)
    ("-500k", "thousand", -500.0),
])
def test_normalize_finance_value(raw, target, expected):
    assert normalize_finance_value(raw, target) == expected

@pytest.mark.asyncio
async def test_check_collection_exists():
    with patch("httpx.AsyncClient.post") as mock_post:
        # Success case
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {"exists": True}
        assert await check_collection_exists("test_col", "http://localhost:8000") is True
        
        # False case
        mock_post.return_value.json.return_value = {"exists": False}
        assert await check_collection_exists("test_col", "http://localhost:8000") is False
        
        # Error case
        mock_post.side_effect = Exception("Connection error")
        assert await check_collection_exists("test_col", "http://localhost:8000") is False

@pytest.mark.asyncio
async def test_get_indexed_paths():
    with patch("httpx.AsyncClient.post") as mock_post:
        # Success case
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {"full_path": ["path1", "path2"]}
        paths = await get_indexed_paths("test_col", "http://localhost:8000")
        assert paths == {"path1", "path2"}
        
        # Error case
        mock_post.side_effect = Exception("Connection error")
        paths = await get_indexed_paths("test_col", "http://localhost:8000")
        assert paths == set()

@pytest.mark.asyncio
async def test_check_project_processing_status():
    # Completely mock the Path objects to avoid real Path behavior
    project_path = MagicMock()
    input_dir = project_path.__truediv__.return_value # project_path / "input"
    input_dir.exists.return_value = True
    
    # Mock files in input directory
    file1 = MagicMock()
    file1.name = "file1.pdf"
    file1.stem = "file1"
    file1.suffix = ".pdf"
    file1.is_file.return_value = True
    file1.relative_to.return_value = MagicMock()
    file1.relative_to.return_value.__str__.return_value = "file1.pdf"
    
    # Setup the parent and div for json_rel_path logic
    # json_rel_path = f.relative_to(input_dir).parent / f"{f.stem}_{f.suffix.lower().lstrip('.')}.json"
    rel_path_mock = file1.relative_to.return_value
    parent_mock = rel_path_mock.parent
    json_file_path_mock = parent_mock.__truediv__.return_value
    
    input_dir.rglob.return_value = [file1]
    
    # processed_dir = project_path / "processed"
    processed_dir = MagicMock()
    # In the code: is_extracted = (processed_dir / json_rel_path).exists()
    extracted_file_mock = processed_dir.__truediv__.return_value
    
    # Update project_path mock to return input_dir or processed_dir
    def project_div_side_effect(name):
        if name == "input": return input_dir
        if name == "processed": return processed_dir
        return MagicMock()
    project_path.__truediv__.side_effect = project_div_side_effect

    with patch("src.common.utils.get_indexed_paths", new_callable=AsyncMock) as mock_get_indexed:
        # Case 1: Everything is done
        mock_get_indexed.return_value = {"file1.pdf"}
        extracted_file_mock.exists.return_value = True
        
        status = await check_project_processing_status(project_path, "col", "url")
        assert status["is_done"] is True
        assert status["total_files"] == 1
        assert status["extracted_files"] == 1
        assert status["indexed_files"] == 1

        # Case 2: Missing extracted
        extracted_file_mock.exists.return_value = False
        status = await check_project_processing_status(project_path, "col", "url")
        assert status["is_done"] is False
        assert status["missing_extracted"] == ["file1.pdf"]

        # Case 3: Missing indexed
        extracted_file_mock.exists.return_value = True
        mock_get_indexed.return_value = set()
        status = await check_project_processing_status(project_path, "col", "url")
        assert status["is_done"] is False
        assert status["missing_indexed"] == ["file1.pdf"]
             
    # Case 4: Input directory doesn't exist
    input_dir.exists.return_value = False
    status = await check_project_processing_status(project_path, "col", "url")
    assert status["is_done"] is False
    assert "error" in status
