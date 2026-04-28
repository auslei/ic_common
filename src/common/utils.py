"""General utility functions for the InvestorIQ project."""

import markdown2
import httpx
from typing import Optional, List, Dict, Any, Set
from pathlib import Path
import re

def sanitize_project_name(name: str) -> str:
    """Sanitize project name for file system and collection naming."""
    # Replace whitespace with underscore
    name = re.sub(r"\s+", "_", name)
    # Remove any character that is not alphanumeric, underscore, or hyphen
    # but keep unicode characters for non-english names
    name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "", name)
    return name

def render_markdown_to_html(markdown_text: str, title: str = "InvestorIQ Documentation") -> str:
    """
    Render markdown text into a full HTML document with basic styling.
    
    Args:
        markdown_text: The markdown content to render.
        title: The HTML page title.
        
    Returns:
        A string containing the full HTML document.
    """
    # Convert markdown to HTML snippet
    # Using extras for better compatibility:
    # - tables: GFM-like tables
    # - fenced-code-blocks: ``` code blocks
    # - code-friendly: prevents _ and __ from being interpreted as italic/bold when in words
    # - task_list: [ ] [x] support
    html_snippet = markdown2.markdown(
        markdown_text, 
        extras=["tables", "fenced-code-blocks", "code-friendly", "task_list", "header-ids"]
    )
    
    # Simple CSS for a clean look
    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        line-height: 1.6;
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
        color: #24292e;
        background-color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        margin-top: 24px;
        margin-bottom: 16px;
        font-weight: 600;
        line-height: 1.25;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 0.3em;
    }
    h1 { font-size: 2em; }
    h2 { font-size: 1.5em; }
    
    pre {
        background-color: #f6f8fa;
        padding: 16px;
        border-radius: 6px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
    }
    code {
        background-color: rgba(27,31,35,0.05);
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 85%;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 16px;
    }
    th, td {
        border: 1px solid #dfe2e5;
        padding: 6px 13px;
    }
    tr:nth-child(even) {
        background-color: #f6f8fa;
    }
    blockquote {
        padding: 0 1em;
        color: #6a737d;
        border-left: 0.25em solid #dfe2e5;
        margin: 0 0 16px 0;
    }
    ul, ol {
        padding-left: 2em;
    }
    li + li {
        margin-top: 0.25em;
    }
    img {
        max-width: 100%;
    }
    hr {
        height: 0.25em;
        padding: 0;
        margin: 24px 0;
        background-color: #e1e4e8;
        border: 0;
    }
    """
    
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="markdown-body">
{html_snippet}
    </div>
</body>
</html>
"""
    return full_html

async def check_collection_exists(collection_name: str, vectordb_url: str) -> bool:
    """Check if a collection exists in VectorDB."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{vectordb_url}/collection_exists", json={"name": collection_name})
            if resp.status_code == 200:
                return resp.json().get("exists", False)
    except Exception:
        pass
    return False

async def get_indexed_paths(collection_name: str, vectordb_url: str) -> Set[str]:
    """Get all unique full_path metadata values indexed in a collection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{vectordb_url}/get_unique_metadata", 
                json={"collection_name": collection_name, "keys": ["full_path"]}
            )
            if resp.status_code == 200:
                return set(resp.json().get("full_path", []))
    except Exception:
        pass
    return set()

async def check_project_processing_status(project_path: Path, collection_name: str, vectordb_url: str) -> Dict[str, Any]:
    """
    Check if all files in project input are extracted and indexed.
    
    Returns:
        Dict with:
            is_done: bool
            total_files: int
            extracted_files: int
            indexed_files: int
            missing_extracted: List[str]
            missing_indexed: List[str]
    """
    input_dir = project_path / "input"
    processed_dir = project_path / "processed"
    
    if not input_dir.exists():
        return {"is_done": False, "error": "Input directory not found", "total_files": 0, "extracted_files": 0, "indexed_files": 0, "missing_extracted": [], "missing_indexed": []}
        
    all_input_files = []
    for f in input_dir.rglob("*"):
        # Skip hidden files and macOS AppleDouble metadata files (e.g. ._filename.ext)
        if f.name.startswith(".") or f.name.startswith("._"):
            continue
        try:
            if f.is_file():
                all_input_files.append(f)
        except PermissionError:
            # Some mounted filesystem metadata entries can be unreadable in containers.
            continue
    indexed_paths = await get_indexed_paths(collection_name, vectordb_url)
    
    status = {
        "is_done": True,
        "total_files": len(all_input_files),
        "extracted_files": 0,
        "indexed_files": 0,
        "missing_extracted": [],
        "missing_indexed": []
    }
    
    if not all_input_files:
        status["is_done"] = False # No files to process is not "done"
        return status
        
    for f in all_input_files:
        rel_path = str(f.relative_to(input_dir))
        full_path = rel_path # Admin UI uses relative path as full_path
        
        # Check extraction
        # json_rel_path matches logic in server.py list_files
        json_rel_path = f.relative_to(input_dir).parent / f"{f.stem}_{f.suffix.lower().lstrip('.')}.json"
        is_extracted = (processed_dir / json_rel_path).exists()
        
        # Check indexing
        is_indexed = full_path in indexed_paths
        
        if is_extracted:
            status["extracted_files"] += 1
        else:
            status["missing_extracted"].append(rel_path)
            status["is_done"] = False
            
        if is_indexed:
            status["indexed_files"] += 1
        else:
            status["missing_indexed"].append(rel_path)
            status["is_done"] = False
            
    return status

import re

def normalize_finance_value(raw_value, target_unit='million'):
    """
    Normalizes mixed-language financial strings into a specific target scale.
    
    Args:
        raw_value (str): The raw string from LLM (e.g., "1.2亿", "$5M", "3500万").
        target_unit (str): The scale to normalize to. 
                           Options: 1, 个, 十, 百, hundred, 千, thousand, 万, million, 亿, billion
    """
    if not raw_value or str(raw_value).lower() == 'null':
        return None

    # 1. Define Multipliers for Input Suffixes
    # These represent the absolute value of 1 unit in that language
    INPUT_MAP = {
        # English
        'k': 1_000,
        'm': 1_000_000,
        'b': 1_000_000_000,
        'bn': 1_000_000_000,
        't': 1_000_000_000_000,
        # Chinese
        '万': 10_000,
        '百万': 1_000_000,
        '千万': 10_000_000,
        '亿': 100_000_000,
    }

    # 2. Define Target Scale Factors
    # This determines the denominator for your final output
    TARGET_MAP = {
        '1': 1, '个': 1,
        '十': 10,
        '百': 100, 'hundred': 100,
        '千': 1_000, 'thousand': 1_000,
        '万': 10_000,
        'million': 1_000_000, 'm': 1_000_000,
        '亿': 100_000_000,
        'billion': 1_000_000_000, 'b': 1_000_000_000
    }

    # Clean the string: Remove commas, currency symbols, and spaces
    clean_str = re.sub(r'[$,￥\s,]', '', str(raw_value)).lower()

    # Regex to split: [Number with decimals] [Optional Units/Suffixes]
    # Matches negative numbers as well
    match = re.match(r"([+-]?\d*\.?\d+)(.*)", clean_str)
    
    if not match:
        return None

    number_part = float(match.group(1))
    suffix_part = match.group(2).strip()

    # Calculate Absolute Value
    # Check for compound Chinese units (like 百万) first by checking longer strings first
    multiplier = 1
    if suffix_part:
        # Check longer matches first (e.g., '百万' before '万')
        sorted_keys = sorted(INPUT_MAP.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in suffix_part:
                multiplier = INPUT_MAP[key]
                break

    absolute_value = number_part * multiplier

    # Apply Target Normalization
    target_factor = TARGET_MAP.get(str(target_unit).lower(), 1_000_000) # Defaults to Million
    
    normalized_value = absolute_value / target_factor
    
    # Return rounded result (4 decimal places is usually enough for VC data)
    return round(normalized_value, 4)

# --- Test Cases ---
# print(f"1.2亿 to Million:  {normalize_finance_value('1.2亿', 'million')}")    # 120.0
# print(f"3500万 to 亿:      {normalize_finance_value('3500万', '万')}")      # 0.35
# print(f"$50M to 1 (Base):  {normalize_finance_value('$50M', '1')}")         # 50000000.0
# print(f"2000 to thousand:  {normalize_finance_value('2000', 'thousand')}")   # 2.0
# print(f"￥100k to 万:      {normalize_finance_value('￥100k', '万')}")      # 10.0