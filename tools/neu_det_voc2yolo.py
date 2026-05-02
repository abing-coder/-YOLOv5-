"""Convert NEU-DET VOC XML annotations to YOLO txt labels in one command.

Usage:
    python tools/neu_det_voc2yolo.py
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def parse_args():
    parser = argparse.ArgumentParser(description="NEU-DET VOC XML -> YOLO txt converter")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("archive/NEU-DET"),
        help="NEU-DET root path containing train/ and validation/",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class order used to map category -> class_id",
    )
    parser.add_argument(
        "--auto-classes",
        action="store_true",
        help="Infer class names from XML automatically (alphabetical order)",
    )
    parser.add_argument(
        "--skip-difficult",
        action="store_true",
        help="Skip objects with difficult=1 in VOC annotation",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing labels directory before conversion",
    )
    parser.add_argument(
        "--write-data-yaml",
        action="store_true",
        default=True,
        help="Write data/neu-det.yaml automatically (default: enabled)",
    )
    return parser.parse_args()


def discover_classes(dataset_root: Path):
    names = set()
    for xml_path in dataset_root.rglob("annotations/*.xml"):
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name:
                names.add(name)
    return sorted(names)


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    xmin = clamp(xmin, 0.0, img_w - 1.0)
    ymin = clamp(ymin, 0.0, img_h - 1.0)
    xmax = clamp(xmax, 0.0, img_w - 1.0)
    ymax = clamp(ymax, 0.0, img_h - 1.0)
    bw = max(xmax - xmin, 1.0)
    bh = max(ymax - ymin, 1.0)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def convert_split(split_root: Path, class_to_id: dict[str, int], skip_difficult: bool, clean: bool):
    ann_dir = split_root / "annotations"
    labels_dir = split_root / "labels"
    images_dir = split_root / "images"
    assert ann_dir.exists(), f"Annotations dir not found: {ann_dir}"
    assert images_dir.exists(), f"Images dir not found: {images_dir}"

    if clean and labels_dir.exists():
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_index = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            image_index.setdefault(p.name, p)
            image_index.setdefault(p.stem, p)

    xml_files = sorted(ann_dir.glob("*.xml"))
    converted = 0
    skipped_obj = 0
    missing_img = 0
    for xml_file in xml_files:
        root = ET.parse(xml_file).getroot()
        filename = (root.findtext("filename") or f"{xml_file.stem}.jpg").strip()
        image_path = image_index.get(filename) or image_index.get(Path(filename).stem)
        if image_path is None or (not image_path.exists()):
            missing_img += 1
            continue

        size_node = root.find("size")
        if size_node is None:
            continue
        img_w = float(size_node.findtext("width", "0"))
        img_h = float(size_node.findtext("height", "0"))
        if img_w <= 0 or img_h <= 0:
            continue

        lines = []
        for obj in root.findall("object"):
            cls_name = (obj.findtext("name") or "").strip()
            if cls_name not in class_to_id:
                skipped_obj += 1
                continue
            difficult = int(obj.findtext("difficult", "0"))
            if skip_difficult and difficult == 1:
                continue

            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            x, y, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"{class_to_id[cls_name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        rel_image_path = image_path.relative_to(images_dir)
        label_path = labels_dir / rel_image_path.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted += 1

    return converted, len(xml_files), skipped_obj, missing_img


def write_data_yaml(repo_root: Path, dataset_root: Path, class_names: list[str]):
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = data_dir / "neu-det.yaml"
    dataset_rel = dataset_root.relative_to(repo_root).as_posix()

    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])
    content = (
        "# Auto-generated by tools/neu_det_voc2yolo.py\n"
        f"path: {dataset_rel}\n"
        "train: train/images\n"
        "val: validation/images\n"
        "test:\n\n"
        "names:\n"
        f"{names_lines}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (args.dataset_root if args.dataset_root.is_absolute() else (repo_root / args.dataset_root)).resolve()
    assert dataset_root.exists(), f"Dataset root not found: {dataset_root}"

    class_names = discover_classes(dataset_root) if args.auto_classes else args.classes
    class_to_id = {name: i for i, name in enumerate(class_names)}

    print(f"Dataset root: {dataset_root}")
    print(f"Class mapping: {class_to_id}")

    for split in ("train", "validation"):
        split_root = dataset_root / split
        converted, total, skipped_obj, missing_img = convert_split(
            split_root=split_root,
            class_to_id=class_to_id,
            skip_difficult=args.skip_difficult,
            clean=args.clean,
        )
        print(
            f"[{split}] converted {converted}/{total} xml files, "
            f"skipped_objects={skipped_obj}, missing_images={missing_img}"
        )

    if args.write_data_yaml:
        yaml_path = write_data_yaml(repo_root=repo_root, dataset_root=dataset_root, class_names=class_names)
        print(f"Generated data config: {yaml_path}")

    print("Done.")


if __name__ == "__main__":
    main()
