"""Mux browser-encoded baseline H.264 into MP4 using only the standard library."""
import base64
import argparse
import json
from pathlib import Path
import struct

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=root / 'h264_samples.json')
parser.add_argument('--output', type=Path, default=root.parents[1] / 'docs/visualizations/neopax_reverse_ad_cost.mp4')
args = parser.parse_args()
data = json.loads(args.input.read_text(encoding='utf-8-sig'))
samples = data['samples']
payloads = [base64.b64decode(sample['data']) for sample in samples]
times = [sample['timestamp'] for sample in samples]
assert times == sorted(set(times)), 'Baseline H.264 must have monotone presentation times'
assert times[0] == 0
durations = [round((times[i + 1] - t) / 1000) if i + 1 < len(times)
             else round((data['duration'] - t) / 1000) for i, t in enumerate(times)]
duration = sum(durations)
width, height = data['width'], data['height']

def pack(fmt, *values):
    return struct.pack('>' + fmt, *values)

def box(kind, *parts):
    body = b''.join(parts)
    return pack('I', 8 + len(body)) + kind.encode('ascii') + body

def full(kind, body, flags=0):
    return box(kind, pack('I', flags), body)

matrix = pack('9I', 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
ftyp = box('ftyp', b'isom', pack('I', 512), b'isomiso2avc1mp41')
mdat = box('mdat', *payloads)
mvhd = full('mvhd', pack('4I', 0, 0, 1000, duration) + pack('IHH', 0x10000, 0x100, 0)
            + b'\0' * 8 + matrix + b'\0' * 24 + pack('I', 2))
tkhd = full('tkhd', pack('5I', 0, 0, 1, 0, duration) + b'\0' * 8
            + pack('4H', 0, 0, 0, 0) + matrix + pack('2I', width << 16, height << 16), flags=7)
mdhd = full('mdhd', pack('4I2H', 0, 0, 1000, duration, 0x55c4, 0))
hdlr = full('hdlr', pack('I', 0) + b'vide' + b'\0' * 12 + b'NEOPAX animation\0')
vmhd = full('vmhd', pack('4H', 0, 0, 0, 0), flags=1)
dinf = box('dinf', full('dref', pack('I', 1) + full('url ', b'', flags=1)))
compressor = b'NEOPAX H.264'
avc1 = box('avc1', b'\0' * 6 + pack('H', 1) + b'\0' * 16
           + pack('2H2I', width, height, 0x480000, 0x480000)
           + pack('IH', 0, 1) + bytes([len(compressor)]) + compressor.ljust(31, b'\0')
           + pack('2H', 24, 0xffff),
           box('avcC', base64.b64decode(data['description'])),
           box('pasp', pack('2I', 1, 1)))
stsd = full('stsd', pack('I', 1) + avc1)
runs = []
for delta in durations:
    if runs and runs[-1][1] == delta:
        runs[-1][0] += 1
    else:
        runs.append([1, delta])
stts = full('stts', pack('I', len(runs)) + b''.join(pack('2I', *run) for run in runs))
stsc = full('stsc', pack('4I', 1, 1, len(samples), 1))
stsz = full('stsz', pack('2I', 0, len(samples)) + b''.join(pack('I', len(p)) for p in payloads))
stco = full('stco', pack('2I', 1, len(ftyp) + 8))
sync_samples = [i + 1 for i, sample in enumerate(samples) if sample['type'] == 'key']
stss = full('stss', pack('I', len(sync_samples)) + b''.join(pack('I', i) for i in sync_samples))
stbl = box('stbl', stsd, stts, stsc, stsz, stco, stss)
minf = box('minf', vmhd, dinf, stbl)
moov = box('moov', mvhd, box('trak', tkhd, box('mdia', mdhd, hdlr, minf)))
out = args.output
out.write_bytes(ftyp + mdat + moov)
assert len(samples) == 45 and duration == 27560 and sync_samples[0] == 1
print(json.dumps({'path': str(out), 'codec': 'H.264 / AVC', 'size': [width, height],
                  'frames': len(samples), 'duration_s': duration / 1000,
                  'bytes': out.stat().st_size}))
