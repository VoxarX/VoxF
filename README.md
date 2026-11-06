# VoxF
Voxar Openings and Endings Fingerprinter | Voxar Fingerprinter

VoxF is a high-performance acoustic fingerprinting engine written in Rust. It is designed to automatically detect opening (OP) and ending (ED) timestamps across anime episodes by identifying structural audio patterns without relying on an external database.

## What It Does
* **Raw Audio Consumption:** Streamlines raw PCM audio input (`.raw`, `s16le` format at 16kHz Mono) into normalized floating-point time data.
* **Short-Time Fourier Transforms (STFT):** Slices continuous audio into overlapping 256ms frames, smooths them with a Hann window function to eliminate spectral leakage, and translates them into a frequency spectrogram using `rustfft`.
* **Constellation Mapping:** Implements a 2D Local Maxima Filter that scans the spectrogram grid, isolates distinct frequency peaks from background noise/dialogue, and maps them as coordinate "stars."
* **Cross-Correlation Detection:** Matches local structural patterns between episodes. Because openings and endings are identical repeated sequences, they produce highly concentrated timing synchronization points (deltas) that instantly reveal where the songs start and stop.

## Current Project State
* [x] **Stage 1:** Binary stream parsing, `i16` to `f32` conversion, and amplitude normalization.
* [x] **Stage 2:** Overlapping frame segmentation and mathematical Hann window edge smoothing.
* [x] **Stage 3:** Forward Fast Fourier Transform processing and raw spectrogram magnitude extraction.
* [x] **Stage 4 (In Progress):** 2D local neighborhood peak filtering complete (extracting over 2.1 million unique audio stars per episode).
* [ ] **Next Up:** Implementing the combinatorial hashing matrix to pair the star points and outputting the final alignment timestamps.

## Future Roadmap (FFmpeg-Style CLI)
The long-term vision is to transition VoxF from a hardcoded single-file pipeline into a versatile, high-utility system tool reminiscent of FFmpeg. Future features will include:
* **Command-Line Interface:** Leveraging crates like `clap` to process explicit dynamic flags (e.g., `voxf -i reference.raw -q query.raw --stride 1024`).
* **Standard I/O Piping:** Integrating `stdin` and `stdout` stream hooks so raw audio data can be piped directly from tools like FFmpeg or `yt-dlp` in real time without generating massive physical intermediary files on your drive.
* **Flexible Export Modality:** Separating application logging metrics onto `stderr` while keeping clean timestamp data matching templates (like JSON or CSV arrays) dedicated to `stdout` for downstream script automation.

## Built With
* [Rust](https://rust-lang.org)
* [rustfft](https://crates.io)
