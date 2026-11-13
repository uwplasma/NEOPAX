"""Package the minimal arrow animation, with no simulation or dependencies added."""
import json
import runpy
import sys
from pathlib import Path
from PIL import Image

base = Path(__file__).resolve().parent
folder = base / 'arrows'
out = base.parents[1] / 'docs/visualizations/neopax_reverse_ad_arrows.gif'
events = json.loads((folder / 'frames.json').read_text(encoding='utf-8-sig'))
frames = []
for event in events:
    with Image.open(folder / f"frame_{event['index']:03d}.png") as image:
        assert image.size == (1280, 588), image.size
        frames.append(image.convert('RGB'))
sample = Image.new('RGB', (1280 * 3, 588), 'white')
for slot, index in enumerate((0, 21, len(frames)-1)):
    sample.paste(frames[index], (slot*1280, 0))
palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
durations = {'start':1600,'forward':320,'objective':1400,'restore':900,
             'replay':480,'adjoint':480,'release':1000,'done':3500}
indexed[0].save(out, save_all=True, append_images=indexed[1:], loop=0,
                duration=[durations[e['kind']] for e in events], optimize=False, disposal=1)
with Image.open(out) as image:
    duration = 0
    for i in range(image.n_frames):
        image.seek(i)
        duration += image.info['duration']
    assert abs(duration / 1000 - 27.56) < 0.001
    assert image.info['loop'] == 0
    print(json.dumps({'gif':str(out),'frames':image.n_frames,'size':image.size,
                      'duration_s':duration/1000,'bytes':out.stat().st_size}))
sys.argv = [str(base / 'mux_mp4.py'), '--input', str(folder / 'h264_samples.json'),
            '--output', str(out.with_suffix('.mp4'))]
runpy.run_path(str(base / 'mux_mp4.py'), run_name='__main__')
