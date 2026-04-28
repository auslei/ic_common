"""Document conversion and file handling utilities."""


import shutil
from pathlib import Path

import pypandoc

from src.common.logger import get_logger

logger = get_logger("document_utils", icon="📄")

def _fix_markdown_lists_and_tables(content: str) -> str:
    """Ensure lists and tables have proper blank lines before them for pandoc compatibility.
    
    Markdown parsers require blank lines before lists and tables. This function adds them
    if missing to prevent them from being ignored during PDF conversion.
    
    Args:
        content: Markdown content as string
        
    Returns:
        str: Fixed markdown content with proper spacing for lists and tables
    """
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect table start (line with | that's not empty)
        is_table_line = '|' in line and stripped and not stripped.startswith('>')
        
        # Detect list start (line starting with -, *, or numbered list)
        is_list_line = stripped and (
            stripped.startswith('- ') or 
            stripped.startswith('* ') or
            (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in '.)')
        )
        
        if (is_table_line or is_list_line) and i > 0:
            prev_line = lines[i - 1].strip()
            # If previous line is not blank and not already part of list/table, add blank line
            if prev_line and not (prev_line.startswith('- ') or prev_line.startswith('* ') or '|' in prev_line):
                fixed_lines.append('')  # Add blank line before list/table
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def md_to_pdf(md_file_path: str, output_path: str) -> None:
    """Convert markdown file to PDF using pandoc with CJK font support.
    
    Automatically fixes common markdown formatting issues (e.g., missing blank lines
    before tables) to ensure proper rendering in PDF.
    
    Args:
        md_file_path: Path to markdown file to convert
        output_path: Path where PDF will be saved
        
    Note: Requires pandoc and CJK fonts. Install via:
        - Ubuntu/Debian: apt-get install pandoc fonts-noto-cjk
        - macOS: brew install pandoc (XeLaTeX required, via MacTeX/BasicTeX)
        - Windows: https://github.com/jgm/pandoc/releases + Noto CJK fonts
    """
    def _find_pdf_engine() -> str:
        # Prefer xelatex (required for reliable CJK font selection). On macOS,
        # TeX binaries are often installed but not present on PATH, so fall back
        # to common install locations.
        candidates = [
            shutil.which("xelatex"),
            "/Library/TeX/texbin/xelatex",
            "/usr/local/texlive/2025/bin/universal-darwin/xelatex",
            "/usr/local/texlive/2024/bin/universal-darwin/xelatex",
        ]
        for c in candidates:
            if not c:
                continue
            p = Path(c)
            if p.exists():
                return str(p)
        return "xelatex"

    def _looks_like_missing_font_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "fontspec" in msg
            or "font" in msg and ("not found" in msg or "cannot be found" in msg or "missing" in msg)
            or "cjkmainfont" in msg
        )

    def _convert_with_cjk_font(md_content: str, cjk_font: str) -> None:
        pdf_engine = _find_pdf_engine()
        # XeLaTeX args with CJK font support and narrow margins.
        # We set both mainfont and CJKmainfont to ensure symbols like ★, ☆, μ
        # are rendered using the comprehensive CJK font instead of falling back
        # to the default LaTeX font (Latin Modern) which lacks them.
        xelatex_args = [
            f"--pdf-engine={pdf_engine}",
            "-V",
            f"mainfont={cjk_font}",
            "-V",
            f"CJKmainfont={cjk_font}",
            "-V",
            "linestretch=1.5",
            "-V",
            "geometry=margin=0.75in",
        ]

        pypandoc.convert_text(
            md_content,
            to="pdf",
            format="md",
            outputfile=output_path,
            extra_args=xelatex_args,
        )

    try:
        md_content = Path(md_file_path).read_text(encoding='utf-8')
        
        # Fix markdown formatting issues before conversion
        md_content = _fix_markdown_lists_and_tables(md_content)

        # Prefer Noto (if installed) but fall back to macOS built-in CJK fonts.
        # This avoids silent PDF failures on macOS when Noto isn't installed.
        cjk_font_candidates = [
            "Noto Sans CJK SC",
            "PingFang SC",
            "Songti SC",
            "Heiti SC",
            "STSong",
            "Arial Unicode MS",
        ]

        last_error: Exception | None = None
        for idx, cjk_font in enumerate(cjk_font_candidates):
            try:
                logger.debug(f"Converting markdown to PDF with CJK font: {cjk_font}")
                _convert_with_cjk_font(md_content, cjk_font)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if idx < len(cjk_font_candidates) - 1 and _looks_like_missing_font_error(e):
                    logger.warning(
                        "PDF conversion failed with font '%s' (will retry with fallback): %s",
                        cjk_font,
                        e,
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error

        logger.debug(f"PDF generated: {output_path}")
        print(f"✓ PDF generated: {output_path}")
    except OSError as e:
        logger.warning(f"Pandoc not installed, skipping PDF generation: {e}")
        print(f"⚠ Pandoc not installed, skipping PDF generation: {e}")
        print(f"  Markdown report saved to: {md_file_path}")
    except Exception as e:
        logger.error(f"Error converting to PDF: {e}")
        print(f"✗ Error converting to PDF: {e}")


if __name__ == "__main__":
    md_to_pdf("src/webservice/README.md", "webservice.pdf")