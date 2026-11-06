use std::fs::File;
use std::io::{BufReader, Read};
use std::f32::consts::PI;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

const WINDOW_SIZE: usize = 4096;
const STRIDE_SIZE: usize = 1024;


struct Peak {
    time_idx: usize,
    freq_idx: usize,
}


fn main() {
    let file = File::open("One-peice-ep-1.raw").expect("Couldnt't find the file");
    let mut reader = BufReader::new(file);
    let mut chunk = [0u8; 2];
    let mut master_audio_data: Vec<f32> = Vec::new();


    loop {
        match reader.read_exact(&mut chunk) {
            
            Ok(_) => {
                // raw file created using "s16le" hence siging the chunk to handle that
                let signed_chunk = i16::from_le_bytes(chunk);
                let f32_chunk = f32::from(signed_chunk);
                master_audio_data.push(f32_chunk / 32768.0); // divide by max i16 range value to scale samples between -1.0 and 1.0.
            }
            Err(_) => {
                println!("Reached the end of the file or encountered an error.");
                break;
            }

        }
    }

    let mut windows: Vec<Vec<f32>> = Vec::new();
    let mut spectrogram: Vec<Vec<f32>> = Vec::new();
    let mut current_index: usize = 0;
    let mut planner: FftPlanner<f32> = FftPlanner::new();
    
    let fft = planner.plan_fft_forward(WINDOW_SIZE);

    while current_index + WINDOW_SIZE <= master_audio_data.len() {
        let end_index = current_index + WINDOW_SIZE;
        let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(WINDOW_SIZE);
        let mut frame: Vec<f32> = master_audio_data[current_index..end_index].to_vec();
        
        // uses Hann's formula to smooth out the sharp edge boundires of the audio slice (does it for each frame) [https://en.wikipedia.org/wiki/Hann_function]
        for i in 0..WINDOW_SIZE {
            let factor = 0.5 * (1.0 - (2.0 * PI * (i as f32) / (WINDOW_SIZE as f32 - 1.0)).cos());
            frame[i] = frame[i]*factor;
            buffer.push(Complex::new(frame[i], 0.0));
        }
        windows.push(frame);
        fft.process(&mut buffer);
        
        let mut frequency_row: Vec<f32> = Vec::with_capacity(WINDOW_SIZE / 2);
        for i in 0..WINDOW_SIZE/2 {
            let loudness = buffer[i].norm();
            frequency_row.push(loudness);
        }
        spectrogram.push(frequency_row);

        current_index += STRIDE_SIZE; 

    }
    
    let mut constellation_map: Vec<Peak> = Vec::new();
    for t in 1..spectrogram.len() - 1 {
        for f in 1..spectrogram[t].len() - 1 {

            let target = spectrogram[t][f];

            // Local Maxima Filter (Dedicides if the segment of audio is peak or background noise)
            if target > spectrogram[t - 1][f - 1]
                && target > spectrogram[t - 1][f]
                && target > spectrogram[t - 1][f + 1]
                && target > spectrogram[t][f - 1]
                && target > spectrogram[t][f + 1]
                && target > spectrogram[t + 1][f - 1]
                && target > spectrogram[t + 1][f]
                && target > spectrogram[t + 1][f + 1]
            {
                constellation_map.push(Peak {
                    time_idx: t,
                    freq_idx: f,
                });
            }
        }
    }
    println!("{}", constellation_map.len());

}