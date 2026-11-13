"""Package browser-rendered schematic frames; no physical calculation."""
import json
from pathlib import Path
from PIL import Image

repo = Path(__file__).resolve().parents[2]
frames_dir = Path(__file__).resolve().parent
metadata = json.loads((frames_dir / "frames.json").read_text(encoding="utf-8-sig"))
images = []
for frame in metadata:
    with Image.open(frames_dir / f"frame_{frame['index']:03d}.png") as image:
        size = (1120, round(image.height * 1120 / image.width))
        images.append(image.convert("RGB").resize(size, Image.Resampling.LANCZOS))

# Share one palette to keep the stationary text and background stable.
sample = Image.new("RGB", (images[0].width * 3, images[0].height))
for slot, index in enumerate((8, 21, len(images) - 1)):
    sample.paste(images[index], (slot * images[0].width, 0))
palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
indexed = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
durations = {"start": 1600, "forward": 320, "objective": 1400,
             "restore": 900, "replay": 480, "adjoint": 480,
             "release": 1000, "done": 3500}
out = repo / "docs/visualizations/neopax_reverse_ad_cost.gif"
indexed[0].save(out, save_all=True, append_images=indexed[1:],
                duration=[durations[frame["kind"]] for frame in metadata],
                loop=0, optimize=False, disposal=1)
with Image.open(out) as image:
    assert image.n_frames == len(metadata)
    assert image.info["loop"] == 0
    duration = 0
    for index in range(image.n_frames):
        image.seek(index)
        duration += image.info["duration"]
    print(json.dumps({"path": str(out), "frames": image.n_frames,
                      "size": image.size, "duration_s": duration / 1000,
                      "bytes": out.stat().st_size}))
