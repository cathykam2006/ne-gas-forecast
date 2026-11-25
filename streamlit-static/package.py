"""
Packages the Streamlit app for the browser (stlite) under site/dist/app/.

Copies app.py, the src/*.py modules and every file in data/processed/ next to
streamlit-static/index.html, and writes manifest.json listing them. Run by
.github/workflows/site.yml after the Observable site is built, so the app at
<site>/app/ always carries the same data as the rest of the site.

    python streamlit-static/package.py site/dist/app
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "site" / "dist" / "app")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)

files = ["app.py"]
files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "src").glob("*.py"))
files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "data" / "processed").iterdir()
                if p.is_file() and not p.name.startswith("."))

for rel in files:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)
shutil.copy2(ROOT / "streamlit-static" / "index.html", OUT / "index.html")
(OUT / "manifest.json").write_text(json.dumps({"files": files}, indent=1))
print(f"packaged {len(files)} files into {OUT}")
