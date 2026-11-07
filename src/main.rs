use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;
use std::env;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::process::{self, Command, Stdio};

const SAMPLE_RATE: f32 = 16_000.0;
const WINDOW_SIZE: usize = 4096;
const STRIDE_SIZE: usize = 1024;

// Bin width = 16000 / 4096 = 3.9 Hz, so 10..1280 is roughly 40 Hz..5 kHz.
const BAND_EDGES: [usize; 8] = [10, 20, 40, 80, 160, 320, 640, 1280];
const PEAK_RATIO: f32 = 4.0; // peak power must beat the frame's mean power by this factor
const POWER_FLOOR: f32 = 1.0; // ignore near-silence

const MIN_TIME_GAP: u32 = 1; // frames
const MAX_TIME_GAP: u32 = 15; // frames (must fit in 4 bits)
const FAN_OUT: usize = 5; // targets paired with each anchor
const MAX_BUCKET: usize = 100; // skip hashes that appear this often in the reference

const MIN_OFFSET_VOTES: u32 = 40; // an offset needs this many votes to count
const MAX_OFFSETS: usize = 8;
const OFFSET_SUPPRESS: i32 = 5; // frames; offsets this close count as the same peak
const BIN_FRAMES: u32 = 16; // ~1 second
const MIN_HITS_PER_BIN: u32 = 3; // hits needed in a 1s bin to call it "matching"
const MAX_GAP_BINS: usize = 4; // bridge up to ~4s of non-matching audio
const MIN_SEGMENT_SECS: f32 = 8.0;

#[derive(Clone, Copy)]
struct Peak {
    time: u32,
    freq: u32,
    power: f32,
}

struct Segment {
    start: f32,     // seconds in the QUERY episode
    end: f32,
    ref_start: f32, // where it starts in the reference episode
    offset: i32,    // reference frame - query frame
    hits: usize,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("fingerprint") if (4..=6).contains(&args.len()) => {
            let number = |i: usize| -> Option<f32> {
                args.get(i).map(|s| {
                    s.parse::<f32>().unwrap_or_else(|_| {
                        eprintln!("'{s}' is not a number");
                        process::exit(1)
                    })
                })
            };
            cmd_fingerprint(&args[2], &args[3], number(4), number(5))
        }
        Some("match") if args.len() == 4 => cmd_match(&args[2], &args[3]),
        Some("compare") if args.len() == 4 => cmd_compare(&args[2], &args[3]),
        _ => {
            eprintln!("usage:");
            eprintln!("  voxf fingerprint <audio> <out.fp> [start_secs] [duration_secs]");
            eprintln!("      save an OP/ED clip's fingerprint (optionally cut from a longer file)");
            eprintln!("  voxf match <episode> <clip.fp>");
            eprintln!("      find that clip inside an episode");
            eprintln!("  voxf compare <reference> <query>");
            eprintln!("      find audio two episodes share");
            eprintln!();
            eprintln!("<audio>/<episode>/<reference>/<query> can be any audio or video file (needs");
            eprintln!("ffmpeg on PATH), or a headerless .raw file (s16le, mono, 16 kHz).");
            process::exit(1);
        }
    }
}

/// Fingerprint one OP/ED clip (the clip must start exactly where the OP/ED starts).
fn cmd_fingerprint(
    clip_path: &str,
    out_path: &str,
    start_secs: Option<f32>,
    duration_secs: Option<f32>,
) {
    let all = load_samples(clip_path);
    let start = ((start_secs.unwrap_or(0.0).max(0.0) * SAMPLE_RATE) as usize).min(all.len());
    let end = match duration_secs {
        Some(d) => (start + (d.max(0.0) * SAMPLE_RATE) as usize).min(all.len()),
        None => all.len(),
    };
    let samples = &all[start..end];
    if samples.len() < WINDOW_SIZE {
        eprintln!("Clip is too short (or the start time is past the end of the file).");
        process::exit(1);
    }
    let peaks = extract_peaks(&samples);
    let mut db = fingerprints(&peaks);
    db.sort_unstable();
    let clip_secs = samples.len() as f32 / SAMPLE_RATE;
    save_db(out_path, clip_secs, &db);
    println!(
        "{} hashes from {:.1}s of audio ({:.1} peaks/s)",
        db.len(),
        clip_secs,
        peaks.len() as f32 / clip_secs.max(1.0)
    );
}

/// Find a fingerprinted clip inside an episode; prints one JSON object.
fn cmd_match(episode_path: &str, db_path: &str) {
    let (clip_secs, db) = load_db(db_path);
    let samples = load_samples(episode_path);
    let episode_secs = samples.len() as f32 / SAMPLE_RATE;
    let peaks = extract_peaks(&samples);
    let segments = match_segments(&db, &peaks);

    match segments.iter().max_by_key(|s| s.hits) {
        None => println!("{{\"found\": false}}"),
        Some(s) => {
            // clip time 0 lines up with episode time -offset (same idea as abs(offset) in Dejavu)
            let start = (-(s.offset as f32) * STRIDE_SIZE as f32 / SAMPLE_RATE).max(0.0);
            let end = (start + clip_secs).min(episode_secs);
            println!(
                "{{\"found\": true, \"start\": {:.2}, \"end\": {:.2}, \"matched_start\": {:.2}, \"matched_end\": {:.2}, \"hits\": {}}}",
                start, end, s.start, s.end, s.hits
            );
        }
    }
}

/// Episode vs episode: list every segment they share.
fn cmd_compare(ref_path: &str, query_path: &str) {
    let (ref_samples, qry_samples) =
        rayon::join(|| load_samples(ref_path), || load_samples(query_path));
    let ref_peaks = extract_peaks(&ref_samples);
    let qry_peaks = extract_peaks(&qry_samples);

    let ref_secs = (ref_samples.len() as f32 / SAMPLE_RATE).max(1.0);
    let qry_secs = (qry_samples.len() as f32 / SAMPLE_RATE).max(1.0);
    println!(
        "peaks/s  reference: {:.1}  query: {:.1}   (aim for tens, not thousands)",
        ref_peaks.len() as f32 / ref_secs,
        qry_peaks.len() as f32 / qry_secs
    );

    let mut db = fingerprints(&ref_peaks);
    db.sort_unstable();
    let segments = match_segments(&db, &qry_peaks);

    if segments.is_empty() {
        println!("No shared segments found.");
        return;
    }
    println!("\nShared segments (query time | reference time):");
    for s in &segments {
        println!(
            "  {} - {}  |  starts at {} in reference  ({} hits)",
            fmt_time(s.start),
            fmt_time(s.end),
            fmt_time(s.ref_start),
            s.hits
        );
    }

    let half = qry_secs / 2.0;
    let longest = |a: &&Segment, b: &&Segment| (a.end - a.start).total_cmp(&(b.end - b.start));
    if let Some(s) = segments.iter().filter(|s| s.start < half).max_by(longest) {
        println!("\nIntro guess: {} - {}", fmt_time(s.start), fmt_time(s.end));
    }
    if let Some(s) = segments.iter().filter(|s| s.start >= half).max_by(longest) {
        println!("Outro guess: {} - {}", fmt_time(s.start), fmt_time(s.end));
    }
}

/// Shared by `match` and `compare`: db must be sorted.
fn match_segments(db: &[(u32, u32)], query_peaks: &[Peak]) -> Vec<Segment> {
    let query_hashes = fingerprints(query_peaks);
    let mut hits = find_hits(db, &query_hashes);
    hits.sort_unstable(); // by offset, then query time
    let offsets = top_offsets(&hits);
    let mut segments: Vec<Segment> = offsets
        .iter()
        .flat_map(|&o| segments_for_offset(&hits, o))
        .collect();


    // Repeating music (choruses, loops) creates weaker "echo" matches inside a real
    // segment. Keep the strongest segments and drop any that mostly overlap one.
    segments.sort_by(|a, b| b.hits.cmp(&a.hits));
    let mut kept: Vec<Segment> = Vec::new();
    for s in segments {
        let len = s.end - s.start;
        let overlapped = kept
            .iter()
            .any(|k| s.end.min(k.end) - s.start.max(k.start) > 0.5 * len);
        if !overlapped {
            kept.push(s);
        }
    }
    kept.sort_by(|a, b| a.start.total_cmp(&b.start));
    kept
}

// .fp file: clip length (f32 LE) followed by sorted (hash, time) pairs (u32 LE each)
fn save_db(path: &str, clip_secs: f32, db: &[(u32, u32)]) {
    let mut bytes = Vec::with_capacity(4 + db.len() * 8);
    bytes.extend_from_slice(&clip_secs.to_le_bytes());
    for &(hash, time) in db {
        bytes.extend_from_slice(&hash.to_le_bytes());
        bytes.extend_from_slice(&time.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|e| {
        eprintln!("Couldn't write {path}: {e}");
        process::exit(1)
    });
}

fn load_db(path: &str) -> (f32, Vec<(u32, u32)>) {
    let bytes = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Couldn't read {path}: {e}");
        process::exit(1)
    });
    if bytes.len() < 4 || (bytes.len() - 4) % 8 != 0 {
        eprintln!("{path} is not a valid .fp file");
        process::exit(1);
    }
    let clip_secs = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let db = bytes[4..]
        .chunks_exact(8)
        .map(|c| {
            (
                u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
            )
        })
        .collect();
    (clip_secs, db)
}

fn fmt_time(secs: f32) -> String {
    format!("{:02}:{:05.2}", (secs / 60.0) as u32, secs % 60.0)
}

/// Centre of the analysis window for a frame, in seconds.
fn frame_to_secs(frame: u32) -> f32 {
    (frame as f32 * STRIDE_SIZE as f32 + WINDOW_SIZE as f32 / 2.0) / SAMPLE_RATE
}

/// .raw files are read as headerless s16le / mono / 16 kHz; anything else goes through ffmpeg.
fn load_samples(path: &str) -> Vec<f32> {
    let is_raw = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map_or(false, |e| e.eq_ignore_ascii_case("raw"));
    let result = if is_raw {
        fs::read(path)
    } else {
        decode_with_ffmpeg(path)
    };
    let bytes = result.unwrap_or_else(|e| {
        eprintln!("Couldn't read {path}: {e}");
        if !is_raw {
            eprintln!("(non-.raw files are decoded with ffmpeg, which must be installed and on your PATH)");
        }
        process::exit(1)
    });
    // s16le, mono, 16 kHz
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

fn decode_with_ffmpeg(path: &str) -> std::io::Result<Vec<u8>> {
    let output = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "ffmpeg could not decode the file",
        ));
    }
    Ok(output.stdout)
}

/// STFT in parallel; only the few peaks per frame are kept, never the spectrogram.
fn extract_peaks(samples: &[f32]) -> Vec<Peak> {
    if samples.len() < WINDOW_SIZE {
        return Vec::new();
    }
    let n_frames = (samples.len() - WINDOW_SIZE) / STRIDE_SIZE + 1;

    // Hann window, computed once.
    let window: Vec<f32> = (0..WINDOW_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f32) / (WINDOW_SIZE as f32)).cos()))
        .collect();

    let fft = FftPlanner::<f32>::new().plan_fft_forward(WINDOW_SIZE);
    let scratch_len = fft.get_inplace_scratch_len();

    let mut peaks: Vec<Peak> = (0..n_frames)
        .into_par_iter()
        .map_init(
            // per-thread buffers, reused for every frame that thread handles
            || {
                (
                    vec![Complex::new(0.0f32, 0.0); WINDOW_SIZE],
                    vec![Complex::new(0.0f32, 0.0); scratch_len],
                )
            },
            |state, t| {
                let (buf, scratch) = state;
                let start = t * STRIDE_SIZE;

                for ((dst, &s), &w) in buf
                    .iter_mut()
                    .zip(&samples[start..start + WINDOW_SIZE])
                    .zip(window.iter())
                {
                    *dst = Complex::new(s * w, 0.0);
                }
                fft.process_with_scratch(&mut buf[..], &mut scratch[..]);

                let lo = BAND_EDGES[0];
                let hi = BAND_EDGES[BAND_EDGES.len() - 1];
                let mean =
                    buf[lo..hi].iter().map(|c| c.norm_sqr()).sum::<f32>() / (hi - lo) as f32;

                let mut out = Vec::with_capacity(BAND_EDGES.len() - 1);
                for band in BAND_EDGES.windows(2) {
                    let mut best_bin = band[0];
                    let mut best_pow = 0.0f32;
                    for k in band[0]..band[1] {
                        let p = buf[k].norm_sqr();
                        if p > best_pow {
                            best_pow = p;
                            best_bin = k;
                        }
                    }
                    if best_pow > PEAK_RATIO * mean && best_pow > POWER_FLOOR {
                        out.push(Peak {
                            time: t as u32,
                            freq: best_bin as u32,
                            power: best_pow,
                        });
                    }
                }
                out
            },
        )
        .flatten()
        .collect();

    peaks.sort_unstable_by_key(|p| (p.time, p.freq));
    peaks
}

/// (hash, anchor_time) pairs. Targets are the strongest peaks in the target zone,
/// not simply the first ones in the list.
fn fingerprints(peaks: &[Peak]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(peaks.len() * FAN_OUT);
    let mut zone_start = 0usize;
    let mut candidates: Vec<&Peak> = Vec::new();

    for anchor in peaks {
        // peaks are sorted by time, so the zone start only ever moves forward
        while zone_start < peaks.len() && peaks[zone_start].time < anchor.time + MIN_TIME_GAP {
            zone_start += 1;
        }
        candidates.clear();
        for p in &peaks[zone_start..] {
            if p.time > anchor.time + MAX_TIME_GAP {
                break;
            }
            candidates.push(p);
        }
        candidates.sort_unstable_by(|a, b| b.power.total_cmp(&a.power));

        for target in candidates.iter().take(FAN_OUT) {
            let dt = target.time - anchor.time;
            // 11 bits | 11 bits | 4 bits
            let hash = (anchor.freq << 15) | (target.freq << 4) | dt;
            out.push((hash, anchor.time));
        }
    }
    out
}

/// For every query hash found in the reference: (ref_time - query_time, query_time).
fn find_hits(db: &[(u32, u32)], query: &[(u32, u32)]) -> Vec<(i32, u32)> {
    let mut hits = Vec::new();
    for &(hash, q_time) in query {
        let lo = db.partition_point(|&(h, _)| h < hash);
        let mut hi = lo;
        while hi < db.len() && db[hi].0 == hash {
            hi += 1;
        }
        if hi == lo || hi - lo > MAX_BUCKET {
            continue;
        }
        for &(_, r_time) in &db[lo..hi] {
            hits.push((r_time as i32 - q_time as i32, q_time));
        }
    }
    hits
}

/// Distinct strong offsets (one per place the shared audio sits differently in the
/// two episodes). Votes from neighbouring offsets are pooled because frame
/// quantisation makes the true offset wobble by +-1.
fn top_offsets(hits: &[(i32, u32)]) -> Vec<i32> {
    let mut counts: Vec<(i32, u32)> = Vec::new();
    for &(o, _) in hits {
        if let Some(last) = counts.last_mut() {
            if last.0 == o {
                last.1 += 1;
                continue;
            }
        }
        counts.push((o, 1));
    }

    let lookup: HashMap<i32, u32> = counts.iter().copied().collect();
    let mut cands: Vec<(i32, u32)> = counts
        .iter()
        .map(|&(o, _)| {
            let votes = (o - 1..=o + 1)
                .map(|x| lookup.get(&x).copied().unwrap_or(0))
                .sum::<u32>();
            (o, votes)
        })
        .collect();
    cands.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let mut chosen: Vec<i32> = Vec::new();
    for (o, votes) in cands {
        if votes < MIN_OFFSET_VOTES || chosen.len() >= MAX_OFFSETS {
            break;
        }
        if chosen.iter().all(|&c| (c - o).abs() > OFFSET_SUPPRESS) {
            chosen.push(o);
        }
    }
    chosen
}

/// Turn the hits at one offset into time segments in the query episode.
fn segments_for_offset(hits: &[(i32, u32)], offset: i32) -> Vec<Segment> {
    // hits are sorted by offset, so the +-1 neighbourhood is one contiguous slice
    let lo = hits.partition_point(|&(o, _)| o < offset - 1);
    let hi = hits.partition_point(|&(o, _)| o <= offset + 1);
    let mut times: Vec<u32> = hits[lo..hi].iter().map(|&(_, t)| t).collect();
    if times.is_empty() {
        return Vec::new();
    }
    times.sort_unstable();

    // ~1 second bins; a bin is "active" if enough hashes agree in it
    let n_bins = (times[times.len() - 1] / BIN_FRAMES) as usize + 1;
    let mut bins = vec![0u32; n_bins];
    for &t in &times {
        bins[(t / BIN_FRAMES) as usize] += 1;
    }

    // merge active bins, bridging short gaps
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut last_active = 0usize;
    for b in 0..=n_bins {
        let active = b < n_bins && bins[b] >= MIN_HITS_PER_BIN;
        if active {
            run_start.get_or_insert(b);
            last_active = b;
        } else if let Some(start) = run_start {
            if b == n_bins || b - last_active > MAX_GAP_BINS {
                runs.push((start, last_active));
                run_start = None;
            }
        }
    }

    let k = MIN_HITS_PER_BIN as usize;
    let mut segments = Vec::new();
    for (first_bin, last_bin) in runs {
        let from = times.partition_point(|&t| t < first_bin as u32 * BIN_FRAMES);
        let to = times.partition_point(|&t| t < (last_bin as u32 + 1) * BIN_FRAMES);
        let run = &times[from..to];
        if run.len() < k {
            continue;
        }

        // trim stray hits: the edge is where k hits first fall inside one bin
        let start = (0..=run.len() - k)
            .find(|&i| run[i + k - 1] < run[i] + BIN_FRAMES)
            .map(|i| run[i])
            .unwrap_or(run[0]);
        let end = (k - 1..run.len())
            .rev()
            .find(|&i| run[i] < run[i + 1 - k] + BIN_FRAMES)
            .map(|i| run[i])
            .unwrap_or(run[run.len() - 1]);

        let start_s = frame_to_secs(start);
        let end_s = frame_to_secs(end);
        if end_s - start_s >= MIN_SEGMENT_SECS {
            segments.push(Segment {
                start: start_s,
                end: end_s,
                ref_start: start_s + offset as f32 * STRIDE_SIZE as f32 / SAMPLE_RATE,
                offset,
                hits: run.len(),
            });
        }
    }
    segments
}