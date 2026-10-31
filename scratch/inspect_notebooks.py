import json
import os
from pathlib import Path

files = [
    r"C:\Users\tusha\Downloads\smart-buildings-probe-ds-k07.ipynb",
    r"C:\Users\tusha\Downloads\smart-buildings-probe-zone-pi-k10.ipynb",
    r"C:\Users\tusha\Downloads\smart-buildings-competition.ipynb",
    r"C:\Users\tusha\Downloads\smart-buildings-zone-pi-hold.ipynb",
    r"C:\Users\tusha\Downloads\smart-buildings-zone-pi-no-night-ceiling.ipynb",
    r"C:\Users\tusha\Downloads\smart-buildings-zone-pi-hold-576.ipynb",
]

out_dir = Path("scratch/extracted_notebooks")
out_dir.mkdir(parents=True, exist_ok=True)

for filepath in files:
    name = Path(filepath).stem
    print("=" * 60)
    print("Inspecting:", name)
    if not os.path.exists(filepath):
        print("Does not exist!")
        continue
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        nb = json.load(f)
    
    code_cells = []
    md_cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            code_cells.append(f"# ===== CELL {i} =====\n" + source)
        elif cell.get("cell_type") == "markdown":
            md_cells.append(f"<!-- ===== MD CELL {i} ===== -->\n" + source)
            
    out_py = out_dir / f"{name}.py"
    with open(out_py, "w", encoding="utf-8") as f:
        f.write("\n\n".join(code_cells))
        
    out_md = out_dir / f"{name}_md.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_cells))
        
    print(f"Extracted {len(code_cells)} code cells -> {out_py}")
    print(f"Extracted {len(md_cells)} markdown cells -> {out_md}")
