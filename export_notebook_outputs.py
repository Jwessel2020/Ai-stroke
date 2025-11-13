from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat


def export_outputs(notebook_path: str | Path, output_path: str | Path) -> Path:
    nb_path = Path(notebook_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)

    with out_path.open("w", encoding="utf-8") as fh:
        for idx, cell in enumerate(nb.cells, start=1):
            if cell.get("cell_type") != "code":
                continue

            fh.write(f"==== Cell {idx} ====\n")
            source = cell.get("source", "").rstrip()
            fh.write("Source:\n")
            fh.write(source + "\n")
            fh.write("Outputs:\n")

            outputs: list[dict[str, Any]] = cell.get("outputs", [])
            if not outputs:
                fh.write("[no output]\n\n")
                continue

            for out_idx, output in enumerate(outputs, start=1):
                output_type = output.get("output_type", "unknown")
                fh.write(f"-- Output {out_idx} ({output_type}) --\n")

                if output_type == "stream":
                    fh.write(output.get("text", ""))
                elif output_type in {"execute_result", "display_data"}:
                    data = output.get("data", {}) or {}
                    text = data.get("text/plain")
                    if text:
                        if isinstance(text, list):
                            text = "".join(text)
                        fh.write(text)
                    else:
                        keys = ", ".join(sorted(data.keys()))
                        fh.write(f"[non-text data: {keys}]\n")
                elif output_type == "error":
                    traceback = output.get("traceback", [])
                    if traceback:
                        fh.write("\n".join(traceback) + "\n")
                    else:
                        fh.write(f"{output.get('ename', '')}: {output.get('evalue', '')}\n")
                else:
                    fh.write(f"[unhandled output type: {output_type}]\n")

            fh.write("\n")

    return out_path


if __name__ == "__main__":
    default_notebook = Path("stroke_modeling.ipynb")
    default_output = Path("logs/stroke_modeling_outputs.txt")
    result_path = export_outputs(default_notebook, default_output)
    print(f"Wrote outputs to {result_path}")
