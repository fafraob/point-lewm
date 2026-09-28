"""Re-encode a matplotlib PDF's rasters through ghostscript.

Shared by the figure scripts, which all have the same problem: their panels are
dense dot patterns (point clouds, attention scatter) that flate cannot compress
at all, so a figure saved at 300 dpi lands in the multi-megabyte range and stays
there. DCT at q=95 takes the same page to a fraction of that with no visible
difference at print size.
"""

import shutil
import subprocess
from pathlib import Path


def compress_pdf(pdf, jpeg_quality=95, dpi=300, log=print):
    """Rewrite `pdf` in place with JPEG-encoded rasters. Returns True if it ran.

    Leaves the original untouched on any failure -- a figure that is too big is
    a nuisance, a figure that is gone is a lost afternoon.
    """
    pdf = Path(pdf)
    gs = shutil.which("gs")
    if gs is None:
        log("[figure] ghostscript not found; leaving the PDF uncompressed")
        return False

    before = pdf.stat().st_size
    tmp = pdf.with_suffix(".compressed.pdf")
    cmd = [gs, "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", "-dNOPAUSE", "-dBATCH",
           "-dQUIET", "-dDownsampleColorImages=true", f"-dColorImageResolution={dpi}",
           "-dColorImageDownsampleType=/Bicubic", "-dAutoFilterColorImages=false",
           # JPEG, not flate: see the module docstring. Text stays vector, and
           # the embedded font is carried through as-is.
           "-dColorImageFilter=/DCTEncode", f"-dJPEGQ={jpeg_quality}",
           f"-sOutputFile={tmp}", str(pdf)]
    if subprocess.run(cmd, check=False).returncode == 0 and tmp.is_file():
        tmp.replace(pdf)
        log(f"[figure] compressed {before / 1e6:.1f} MB -> {pdf.stat().st_size / 1e6:.1f} MB")
        return True
    tmp.unlink(missing_ok=True)
    log("[figure] ghostscript pass failed; keeping the original PDF")
    return False
