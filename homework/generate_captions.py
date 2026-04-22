import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info
from .supertux_parse import generate_caption_lines, load_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list[str]:
    """
    Generate caption lines for a specific view (one string per fact).
    """
    return generate_caption_lines(info_path, view_index, img_width, img_height)


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def build_train_captions(
    data_root: str = "data",
    out_name: str = "supertux_captions.json",
    split: str = "train",
):
    root = Path(data_root) / split
    out_path = root / out_name
    rows: list[dict] = []
    for info_path in sorted(root.glob("*_info.json")):
        stem = info_path.stem.replace("_info", "")
        info = load_info(str(info_path))
        n_views = len(info.get("detections", []))
        for view_index in range(n_views):
            img_path = root / f"{stem}_{view_index:02d}_im.jpg"
            if not img_path.exists():
                continue
            image_rel = f"{split}/{stem}_{view_index:02d}_im.jpg"
            for caption in generate_caption(str(info_path), view_index):
                rows.append({"image_file": image_rel, "caption": caption})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} caption rows to {out_path}")


def main():
    fire.Fire({"check": check_caption, "build_train_captions": build_train_captions})


if __name__ == "__main__":
    main()
