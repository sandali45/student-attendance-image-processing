"""One-command runner for Member 5 (signature-box extraction).

Just run:

    python run_signature_extraction.py

It reads every corrected sheet in output/corrected_images/, runs the
earlier members' steps (preprocess -> threshold -> table detection) to
get the table layout, then extracts one cleaned signature box per
student row and saves them into output/signature_regions/.
"""

import os
import sys

# make the src/ modules importable no matter where we're launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from signature_extraction import run  # noqa: E402


if __name__ == "__main__":
    # optional: pass specific sheet numbers, e.g. python run_signature_extraction.py 1 3
    sheet_names = sys.argv[1:] if len(sys.argv) > 1 else None
    run(sheet_names)
