#!/usr/bin/env python3
"""Filter AlphaFold DB structures by residue-mean pLDDT and max PAE."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import multiprocessing
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from Bio.PDB import MMCIFParser

LOGGER = logging.getLogger("afdb_filter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter AlphaFold DB structures by residue-mean pLDDT (from B-factor) "
            "and max PAE (from JSON predicted_aligned_error/pae)."
        )
    )
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/sdb/haoran/afdb_web"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/sdb/haoran/afdb_web_filtered"))
    parser.add_argument("--plddt-threshold", type=float, default=70.0, help="Minimum residue-mean pLDDT")
    parser.add_argument("--pae-threshold", type=float, default=10.0, help="Maximum max-PAE")
    parser.add_argument("--csv-name", type=str, default="filtered_list.csv")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, multiprocessing.cpu_count())),
        help="Number of parallel worker processes (default: min(8, CPU cores)).",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s [%(levelname)s] %(message)s")


def iter_structure_files(root: Path) -> Iterator[Path]:
    patterns = ("**/*.pdb", "**/*.cif", "**/*.mmcif", "**/*.pdb.gz", "**/*.cif.gz", "**/*.mmcif.gz")
    for pattern in patterns:
        yield from (p for p in root.glob(pattern) if p.is_file())


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _structure_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".pdb.gz"):
        return name[: -len(".pdb.gz")]
    if name.endswith(".cif.gz"):
        return name[: -len(".cif.gz")]
    if name.endswith(".mmcif.gz"):
        return name[: -len(".mmcif.gz")]
    return path.stem


def extract_mean_plddt_pdb_residue(structure_path: Path) -> float:
    """Fast residue-mean pLDDT extraction from PDB/PDB.GZ B-factor columns."""
    residue_sums: dict[tuple[str, str, str, str], tuple[float, int]] = {}

    with _open_text(structure_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 66:
                continue
            try:
                bfactor = float(line[60:66])
            except ValueError:
                continue

            chain_id = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            altloc = line[16:17].strip()
            key = (chain_id, resseq, icode, altloc)

            prev_sum, prev_count = residue_sums.get(key, (0.0, 0))
            residue_sums[key] = (prev_sum + bfactor, prev_count + 1)

    if not residue_sums:
        raise ValueError("No B-factor values found in ATOM/HETATM records.")

    residue_means = [s / c for s, c in residue_sums.values() if c > 0]
    return sum(residue_means) / len(residue_means)


def extract_mean_plddt(structure_path: Path) -> float:
    name = structure_path.name.lower()
    if name.endswith(".pdb") or name.endswith(".pdb.gz"):
        return extract_mean_plddt_pdb_residue(structure_path)

    parser = MMCIFParser(QUIET=True)

    if name.endswith(".gz"):
        with _open_text(structure_path) as handle:
            content = handle.read()
        with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            structure = parser.get_structure(_structure_stem(structure_path), tmp.name)
    else:
        structure = parser.get_structure(_structure_stem(structure_path), str(structure_path))

    residue_means: list[float] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                atom_bfactors = [atom.get_bfactor() for atom in residue.get_atoms()]
                if atom_bfactors:
                    residue_means.append(sum(atom_bfactors) / len(atom_bfactors))

    if not residue_means:
        raise ValueError("No residues with B-factor values found.")

    return sum(residue_means) / len(residue_means)


def _resolve_json_content(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    raise ValueError("Unsupported JSON format (expected object or list[object]).")


def _extract_pae_matrix(payload: dict) -> list[list[float]]:
    matrix = payload.get("predicted_aligned_error")
    if matrix is None:
        pae_value = payload.get("pae")
        if isinstance(pae_value, dict):
            matrix = pae_value.get("matrix")
        else:
            matrix = pae_value
    if matrix is None:
        raise KeyError("No PAE matrix found (expected predicted_aligned_error, pae, or pae.matrix).")
    return matrix


def extract_max_pae(json_path: Path) -> float:
    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    payload = _resolve_json_content(raw)
    matrix = _extract_pae_matrix(payload)

    max_value: float | None = None
    for row in matrix:
        for value in row:
            v = float(value)
            if max_value is None or v > max_value:
                max_value = v

    if max_value is None:
        raise ValueError("PAE matrix is empty.")

    return max_value


def find_pae_json(structure_path: Path) -> Path | None:
    """Prefer predicted_aligned_error JSON files, with safe fallbacks."""
    parent = structure_path.parent
    stem = _structure_stem(structure_path)

    keyword_candidates = sorted(parent.glob("*predicted_aligned_error*.json"))
    if len(keyword_candidates) == 1:
        return keyword_candidates[0]

    mapped = stem.replace("-model_", "-predicted_aligned_error_").replace(
        "-model_v", "-predicted_aligned_error_v"
    )
    mapped_file = parent / f"{mapped}.json"
    if mapped_file.exists():
        return mapped_file

    stem_keyword_candidates = sorted(parent.glob(f"{stem}*predicted_aligned_error*.json"))
    if stem_keyword_candidates:
        return stem_keyword_candidates[0]

    generic_pae = sorted(parent.glob("*pae*.json"))
    if len(generic_pae) == 1:
        return generic_pae[0]

    return None


def process_file(
    structure_path: Path,
    input_root: Path,
    output_root: Path,
    plddt_threshold: float,
    pae_threshold: float,
) -> dict | None:
    json_path = find_pae_json(structure_path)
    if json_path is None:
        LOGGER.warning("SKIP missing JSON for: %s", structure_path)
        return None

    try:
        mean_plddt = extract_mean_plddt(structure_path)
        max_pae = extract_max_pae(json_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("SKIP parse error for %s (%s)", structure_path, exc)
        return None

    protein_id = _structure_stem(structure_path)
    rel_species_dir = structure_path.parent.relative_to(input_root)
    species = str(rel_species_dir)

    if mean_plddt < plddt_threshold or max_pae > pae_threshold:
        LOGGER.info(
            "FAIL %s | species=%s | mean_pLDDT=%.2f | max_PAE=%.2f",
            protein_id,
            species,
            mean_plddt,
            max_pae,
        )
        return None

    out_dir = output_root / rel_species_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_structure = out_dir / structure_path.name
    out_json = out_dir / json_path.name
    shutil.copy2(structure_path, out_structure)
    shutil.copy2(json_path, out_json)

    LOGGER.info(
        "PASS %s | species=%s | mean_pLDDT=%.2f | max_PAE=%.2f",
        protein_id,
        species,
        mean_plddt,
        max_pae,
    )
    return {
        "protein_id": protein_id,
        "species": species,
        "mean_pLDDT": f"{mean_plddt:.4f}",
        "max_PAE": f"{max_pae:.4f}",
        "original_path": str(structure_path),
        "output_path": str(out_structure),
    }


def _process_file_star(args: tuple[Path, Path, Path, float, float]) -> dict | None:
    return process_file(*args)


def _iter_job_args(
    input_root: Path,
    output_root: Path,
    plddt_threshold: float,
    pae_threshold: float,
) -> Iterable[tuple[Path, Path, Path, float, float]]:
    for structure_path in iter_structure_files(input_root):
        yield (structure_path, input_root, output_root, plddt_threshold, pae_threshold)


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["protein_id", "species", "mean_pLDDT", "max_PAE", "original_path", "output_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found or not directory: {input_root}")

    LOGGER.info("Input root: %s", input_root)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Thresholds: mean_pLDDT >= %.2f, max_PAE <= %.2f", args.plddt_threshold, args.pae_threshold)

    rows: list[dict] = []
    processed = 0

    if args.workers <= 1:
        for job in _iter_job_args(input_root, output_root, args.plddt_threshold, args.pae_threshold):
            processed += 1
            result = process_file(*job)
            if result is not None:
                rows.append(result)
            if processed % 1000 == 0:
                LOGGER.info("Processed %d structures", processed)
    else:
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = pool.imap_unordered(
                _process_file_star,
                _iter_job_args(input_root, output_root, args.plddt_threshold, args.pae_threshold),
                chunksize=50,
            )
            for result in results:
                processed += 1
                if result is not None:
                    rows.append(result)
                if processed % 1000 == 0:
                    LOGGER.info("Processed %d structures", processed)

    csv_path = output_root / args.csv_name
    write_csv(rows, csv_path)
    LOGGER.info("Processed structures: %d", processed)
    LOGGER.info("Passed structures: %d", len(rows))
    LOGGER.info("CSV written to: %s", csv_path)


if __name__ == "__main__":
    main()
