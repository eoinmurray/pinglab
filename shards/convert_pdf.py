#!/usr/bin/env python3
"""Convert PDF to HTML with local images using Mathpix.

This is a standalone script for PDF conversion that was previously
part of the main pinglab CLI.

Usage:
    python src/scripts/convert_pdf.py path/to/file.pdf

    # Or make executable and run directly:
    chmod +x src/scripts/convert_pdf.py
    ./src/scripts/convert_pdf.py path/to/file.pdf
"""

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


def download_mathpix_images(html_file: Path, output_dir: Path | None = None):
    """Download all Mathpix CDN images from an HTML file and rewrite URLs to local paths.

    Args:
        html_file: Path to HTML file with Mathpix CDN images
        output_dir: Directory to save images (default: {html_file}_images/)
    """
    if output_dir is None:
        output_dir = html_file.parent / f"{html_file.stem}_images"

    output_dir.mkdir(exist_ok=True)

    # Read HTML content
    content = html_file.read_text(encoding="utf-8")

    # Find all Mathpix CDN image URLs (with or without quotes)
    # Matches both: src="URL" and src=URL
    pattern = r'src=["\']{0,1}(https://cdn\.mathpix\.com/[^"\'\s>]+)["\']{0,1}'
    matches = re.findall(pattern, content)

    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for url in matches:
        if url not in seen:
            seen.add(url)
            unique_matches.append(url)
    matches = unique_matches

    print(f"Found {len(matches)} images to download")

    # Download each image
    for i, url in enumerate(matches, 1):
        # Create filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = ".jpg" if "jpg" in url else ".png"
        filename = f"img_{i:03d}_{url_hash}{ext}"
        filepath = output_dir / filename

        print(f"  [{i}/{len(matches)}] Downloading {filename}...")
        urlretrieve(url, filepath)

        # Replace URL in content with local path (handle both quoted and unquoted)
        local_path = f"{output_dir.name}/{filename}"
        # Replace all variations: src="URL", src='URL', and src=URL
        content = content.replace(f'src="{url}"', f'src="{local_path}"')
        content = content.replace(f"src='{url}'", f"src='{local_path}'")
        content = content.replace(f"src={url}", f'src="{local_path}"')

    # Write updated HTML
    output_file = html_file.parent / f"{html_file.stem}_local{html_file.suffix}"
    output_file.write_text(content, encoding="utf-8")

    print(f"\n✅ Downloaded {len(matches)} images to: {output_dir}")
    print(f"✅ Created local HTML: {output_file}")


def convert_pdf(pdf_path: Path):
    """Extract PDF to HTML with local images using Mathpix.

    Args:
        pdf_path: Path to PDF file to convert
    """
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"Error: File must be a PDF: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Get output directory
    output_dir = pdf_path.parent / pdf_path.stem

    print(f"📄 Extracting PDF: {pdf_path}")
    print(f"   Output: {output_dir}/")
    print("")

    # Create temp directory for Mathpix output
    with tempfile.TemporaryDirectory() as tmpdir:
        html_file = Path(tmpdir) / f"{pdf_path.stem}.html"

        # Step 1: Convert PDF to HTML with Mathpix
        print("Step 1/2: Converting PDF with Mathpix...")
        result = subprocess.run(
            [
                "mathpix",
                "convert",
                str(pdf_path),
                str(html_file),
                "--formats",
                "html",
            ],
            check=False,
        )

        if result.returncode != 0:
            print("Error: Mathpix conversion failed", file=sys.stderr)
            print("Make sure Mathpix CLI is installed and configured", file=sys.stderr)
            sys.exit(1)

        if not html_file.exists():
            print(f"Error: HTML file not created: {html_file}", file=sys.stderr)
            sys.exit(1)

        # Step 2: Download images locally
        print("Step 2/2: Downloading images...")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            download_mathpix_images(html_file=html_file, output_dir=output_dir)
        except Exception as e:
            print(f"Error downloading images: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\n✅ PDF extracted to: {output_dir}/")


def main():
    """Main entry point for CLI."""
    if len(sys.argv) != 2:
        print("Usage: convert_pdf.py <path-to-pdf>")
        print("")
        print("Extract PDF to HTML with local images using Mathpix.")
        print("")
        print("Example:")
        print("  python src/scripts/convert_pdf.py paper.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    convert_pdf(pdf_path)


if __name__ == "__main__":
    main()
