
# VoxF

Audio fingerprinter that finds anime openings and endings. It matches audio, so it works
on any show where the OP/ED is the same recording from episode to episode.

Single Rust binary. Needs [ffmpeg](https://ffmpeg.org/) on your PATH to read normal
audio/video files (not needed if you only feed it `.raw` files).

## Build

```
cargo build --release
```

The binary is `target/release/voxf` (`voxf.exe` on Windows). Always use `--release`;
debug builds are much slower.

## Usage

Any audio or video file works as input (`.mkv`, `.mp4`, `.mp3`, ...). A headerless
`.raw` file (s16le, mono, 16 kHz) is read directly.

### 1. Find the OP/ED by comparing two episodes

```
voxf compare ep01.mkv ep03.mkv
```

Lists every segment the two episodes share, with times in the *second* episode, plus an
"Intro guess" and an "Outro guess".

### 2. Fingerprint a clean clip once

Cut the OP (or ED) out of an episode using the times from step 1. The clip must start
exactly where the OP/ED starts.

```
voxf fingerprint ep03.mkv op.fp 0 109        # start at 0 s, 109 s long
voxf fingerprint ep03.mkv ed.fp 1375         # from 1375 s to the end of the file
```

The optional `start` and `duration` are in seconds. The tool prints peaks/s; aim for
tens (roughly 40-100). Hundreds or thousands means the peak threshold is too low.

### 3. Match every other episode

```
voxf match ep05.mkv op.fp
voxf match ep05.mkv ed.fp
```

Prints one JSON line:

```
{"found": true, "start": 0.00, "end": 109.00, "matched_start": 0.13, "matched_end": 109.25, "hits": 48493}
```

- `start` / `end`: where the whole clip sits in the episode (start = where the clip's
  time 0 lands; end = start + clip length).
- `matched_start` / `matched_end`: the span that actually matched. If the episode only
  contains part of the clip, trust these.
- `{"found": false}`: the clip isn't in this episode.

PowerShell loop over a folder:

```
Get-ChildItem *.mkv | ForEach-Object { "$($_.Name): " + (.\voxf.exe match $_.FullName op.fp) }
```

## How it works

1. Decode to mono 16 kHz, then a 4096-sample Hann-windowed FFT every 1024 samples (64 ms).
2. Keep only the strongest bin in each of 7 log-spaced frequency bands per frame, if it
   stands well above the frame's average (about 60-70 peaks/s).
3. Pair each peak with the 5 strongest peaks in the next ~1 s and hash
   (freq A, freq B, time gap) into a `u32`.
4. Look up the episode's hashes in the clip's hashes and vote on the time offset between
   them. A real match makes thousands of hashes agree on one offset.
5. Pool neighbouring offsets, take every strong offset, turn the hits at each into time
   segments (1 s bins, short gaps bridged), and drop weak echoes of stronger segments.

## Accuracy

Measured on One Piece episodes 1 and 3, whose shared audio I checked directly by
comparing the waveforms:

|       | Waveform            | VoxF                |
| ----- | ------------------- | ------------------- |
| Intro | 0 - ~109.3 s        | 0.13 - 109.25 s     |
| Outro | ~1375.0 - ~1450.3 s | 1375.23 - 1449.98 s |

About +-0.25 s. Not yet tested on other shows.

## Tuning

Constants at the top of `src/main.rs`:

| Constant             | Raise it if...                      | Lower it if...                  |
| -------------------- | ----------------------------------- | ------------------------------- |
| `PEAK_RATIO`       | too many peaks/s, noisy matches     | too few peaks, matches missing  |
| `MIN_OFFSET_VOTES` | false segments appear               | a real short segment is missed  |
| `MIN_HITS_PER_BIN` | segments run on past their real end | segments come out in pieces     |
| `MAX_GAP_BINS`     | one segment is split in two         | two segments get merged         |
| `MIN_SEGMENT_SECS` | short jingles show up               | short real segments are dropped |

## Limitations

- Needs the audio to be the same recording in both places. A different dub, voices over
  the OP, or a re-pitched version can break matching.
- The clip must start exactly where the OP/ED starts for `start`/`end` to be right.
- Times are accurate to about a quarter second (64 ms hops, 256 ms windows).
- Whole files are loaded into memory (about 50 MB per 25-minute episode).
