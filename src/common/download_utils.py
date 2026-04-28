import os
from pathlib import Path

from src.common.logger import get_logger

logger = get_logger("download_utils", icon="⬇️")

def download_media(url: str, output_dir: str | Path) -> Path:
    """
    Download media from a URL using yt-dlp to a specified directory.
    If the file already exists, avoids re-downloading.
    Returns the local Path to the downloaded (or existing) file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a standard output template recognizable by yt-dlp.
    # We prioritize a simple filename to avoid massive URL hashes, 
    # but yt-dlp will handle deduplication/extensions.
    ydl_opts = {
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best', # Download best available quality
        'quiet': True,
        'no_warnings': True,
        # Restrict filenames to avoid weird characters from URL titles breaking the OS
        'restrictfilenames': True, 
        'nooverwrites': True, # Do not overwrite existing files
    }

    logger.info(f"Downloading media from URL: {url} to {output_dir}")

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get the final resolved filename without downloading yet
            info = ydl.extract_info(url, download=False)
            
            if info is None:
                raise RuntimeError(f"Failed to extract info from {url}")

            # If it's a playlist or multiple files, we take the first one for simplicity right now
            if 'entries' in info:
                entries = [e for e in info['entries'] if e is not None]
                if not entries:
                     raise RuntimeError(f"No valid entries found for playlist URL: {url}")
                info = entries[0]
            
            # Predict the final local file path
            expected_filename = ydl.prepare_filename(info)
            local_path = Path(expected_filename)
            
            # If the file already exists on disk, skip downloading
            if local_path.exists():
                logger.info(f"File already exists in cache, skipping download: {local_path}")
                return local_path

            # Otherwise, trigger the actual download
            logger.info("File not found in cache. Starting download...")
            ydl.download([url])
            
            if not local_path.exists():
                raise FileNotFoundError(f"yt-dlp claimed to download to {local_path}, but file is missing.")
            
            logger.info(f"Successfully downloaded to: {local_path}")
            return local_path
            
    except Exception as e:
        logger.error(f"Failed to download media from {url}: {e}")
        raise
