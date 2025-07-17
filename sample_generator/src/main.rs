use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use regex::Regex;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crossbeam::channel;
use parking_lot::Mutex;

const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const MIN_LEN: usize = 2;
const MAX_LEN: usize = 100;
const ERRORS_PER_SAMPLE: usize = 2;
const CHUNK_SIZE: usize = 10_000;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file with sentences
    #[arg(short, long)]
    input: PathBuf,

    /// Output file for samples
    #[arg(short, long)]
    output: PathBuf,

    /// Probability of selecting each sentence (0.0 to 1.0)
    #[arg(short, long, default_value_t = 1.0)]
    selection_prob: f64,
}

#[derive(Debug)]
struct Sample {
    context: Vec<String>,
    misspelled: String,
    target: String,
}

fn smart_tokenize(text: &str) -> Vec<String> {
    lazy_static::lazy_static! {
        static ref RE: Regex = Regex::new(r"\w+(?:'\w+)?|[^\w\s]").unwrap();
    }
    RE.find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

fn random_corrupt(word: &str) -> String {
    let mut rng = thread_rng();
    
    // 20% chance to return unchanged
    if rng.gen::<f64>() < 0.2 {
        return word.to_string();
    }

    let word_bytes = word.as_bytes();
    if word_bytes.is_empty() {
        return word.to_string();
    }

    match rng.gen_range(0..4) {
        0 => { // insert
            let pos = rng.gen_range(0..=word_bytes.len());
            let c = ALPHABET[rng.gen_range(0..ALPHABET.len())] as char;
            let mut result = word[..pos].to_string();
            result.push(c);
            result.push_str(&word[pos..]);
            result
        },
        1 => { // delete
            if word_bytes.len() > 1 {
                let pos = rng.gen_range(0..word_bytes.len());
                let mut result = word[..pos].to_string();
                result.push_str(&word[pos + 1..]);
                result
            } else {
                word.to_string()
            }
        },
        2 => { // substitute
            let pos = rng.gen_range(0..word_bytes.len());
            let mut alphabet: Vec<u8> = ALPHABET.to_vec();
            alphabet.retain(|&x| x != word_bytes[pos]);
            let c = alphabet[rng.gen_range(0..alphabet.len())] as char;
            let mut result = word[..pos].to_string();
            result.push(c);
            result.push_str(&word[pos + 1..]);
            result
        },
        3 => { // transpose
            if word_bytes.len() > 1 {
                let pos = rng.gen_range(0..word_bytes.len() - 1);
                let mut result = word[..pos].to_string();
                result.push(word.chars().nth(pos + 1).unwrap());
                result.push(word.chars().nth(pos).unwrap());
                result.push_str(&word[pos + 2..]);
                result
            } else {
                word.to_string()
            }
        },
        _ => unreachable!(),
    }
}

fn generate_error_samples(tokens: &[String]) -> Vec<Sample> {
    if tokens.len() < MIN_LEN || tokens.len() > MAX_LEN {
        return vec![];
    }

    let mut samples = Vec::new();
    let max_target_pos = tokens.len().min(11);
    let mut rng = thread_rng();

    for i in (MIN_LEN-1)..max_target_pos {
        let target_word = &tokens[i];
        if target_word.len() < 2 {
            continue;
        }

        // Choose 2 random context lengths
        let max_ctx = i.min(10);
        let mut possible_lengths: Vec<usize> = (0..=max_ctx).collect();
        let context_lengths = if possible_lengths.len() > 2 {
            possible_lengths.shuffle(&mut rng);
            &possible_lengths[0..2]
        } else {
            &possible_lengths
        };

        // Generate samples for each context length
        for &ctx_len in context_lengths {
            let start_idx = i.saturating_sub(ctx_len);
            let context: Vec<String> = tokens[start_idx..i].to_vec();
            
            let mut used = HashSet::new();
            let mut error_samples = Vec::new();
            let mut attempts = 0;

            // Generate error samples
            while error_samples.len() < ERRORS_PER_SAMPLE && attempts < 20 {
                let corrupted = random_corrupt(target_word);
                attempts += 1;

                if !used.contains(&corrupted) {
                    used.insert(corrupted.clone());
                    error_samples.push(Sample {
                        context: context.clone(),
                        misspelled: corrupted,
                        target: target_word.clone(),
                    });
                }
            }

            // Fill remaining samples if needed
            while error_samples.len() < ERRORS_PER_SAMPLE {
                let corrupted = random_corrupt(target_word);
                error_samples.push(Sample {
                    context: context.clone(),
                    misspelled: corrupted,
                    target: target_word.clone(),
                });
            }

            samples.extend(error_samples);
        }
    }

    samples
}

fn process_chunk(chunk: Vec<String>, selection_prob: f64) -> Vec<Sample> {
    let mut rng = thread_rng();
    chunk.into_iter()
        .filter(|line| !line.trim().is_empty())
        .filter(|_| rng.gen::<f64>() <= selection_prob)
        .flat_map(|line| {
            let tokens = smart_tokenize(&line);
            generate_error_samples(&tokens)
        })
        .collect()
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    
    // Set up progress tracking
    let total_lines = BufReader::new(File::open(&args.input)?).lines().count();
    let pb = ProgressBar::new(total_lines as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("##-"));

    // Set up parallel processing channels
    let (tx, rx) = channel::bounded(CHUNK_SIZE);
    let processed_samples = Arc::new(AtomicUsize::new(0));
    let processed_samples_clone = processed_samples.clone();

    // Spawn writer thread
    let writer_handle = std::thread::spawn(move || {
        let mut writer = BufWriter::new(File::create(args.output).unwrap());
        let mut total_samples = 0;

        while let Ok(samples) = rx.recv() {
            if samples.is_empty() {
                break;
            }

            // Write samples in batches
            for sample in samples {
                writeln!(writer, "{}\t{}\t{}", 
                    sample.context.join(" "),
                    sample.misspelled,
                    sample.target
                ).unwrap();
                total_samples += 1;
            }

            processed_samples.fetch_add(1, Ordering::SeqCst);
        }

        total_samples
    });

    // Process chunks in parallel
    let chunk_size = CHUNK_SIZE;
    let mut current_chunk = Vec::with_capacity(chunk_size);
    
    let file = File::open(&args.input)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        current_chunk.push(line);

        if current_chunk.len() >= chunk_size {
            let chunk = std::mem::take(&mut current_chunk);
            current_chunk = Vec::with_capacity(chunk_size);

            // Process chunk in parallel
            let samples = process_chunk(chunk, args.selection_prob);
            tx.send(samples).unwrap();

            pb.inc(chunk_size as u64);
            pb.set_message(format!("Samples: {}", processed_samples_clone.load(Ordering::SeqCst)));
        }
    }

    // Process remaining chunk
    if !current_chunk.is_empty() {
        let samples = process_chunk(current_chunk, args.selection_prob);
        tx.send(samples).unwrap();
    }

    // Signal completion
    drop(tx);
    
    // Wait for writer to finish
    let total_samples = writer_handle.join().unwrap();
    pb.finish_with_message(format!("Generated {} samples", total_samples));

    Ok(())
}
