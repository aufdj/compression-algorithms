pub mod bufio;
pub mod lz;
pub mod ari;
pub mod huffman;
pub mod bwt;

use std::fs::metadata;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

fn main() {
    let time = Instant::now();
    let args = std::env::args().skip(1).collect::<Vec<String>>();
    if args.len() != 4 {
        print_usage();
    }

    let algorithm = args[0].as_str();
    let mode = args[1].as_str();
    let file_in_str = &args[2];
    let file_out_str = &args[3];

    let file_in = BufReader::with_capacity(
        1 << 20, 
        File::open(file_in_str)
        .unwrap_or_else(|_| panic!("Could not open input file {}\n", &file_in_str))
    );

    let file_out = BufWriter::with_capacity(
        1 << 20, 
        File::create(file_out_str)
        .unwrap_or_else(|_| panic!("Could not open output file {}\n", &file_out_str))
    );

    match (algorithm, mode) {
        ("-lz77", "-c") => { 
            crate::lz::lz77::Lz77::new(file_in, file_out).compress(); 
        }
        ("-lz77", "-d") => { 
            crate::lz::lz77::Lz77::new(file_in, file_out).decompress(); 
        }
        ("-lzw", "-c") => { 
            crate::lz::lzw::lzw_compress(file_in, file_out); 
        }
        ("-lzw", "-d") => { 
            crate::lz::lzw::lzw_decompress(file_in, file_out); 
        }
        ("-flzp", "-c") => { 
            crate::lz::flzp::flzp_compress(file_in, file_out); 
        }
        ("-flzp", "-d") => { 
            crate::lz::flzp::flzp_decompress(file_in, file_out); 
        }
        ("-fpaq", "-c") => { 
            crate::ari::fpaq::fpaq_compress(file_in, file_out); 
        }
        ("-fpaq", "-d") => { 
            crate::ari::fpaq::fpaq_decompress(file_in, file_out); 
        }
        ("-lpaq1", "-c") => { 
            crate::ari::lpaq1::lpaq1_compress(file_in, file_out); 
        }
        ("-lpaq1", "-d") => { 
            crate::ari::lpaq1::lpaq1_decompress(file_in, file_out); 
        }
        ("-huffman", "-c") => { 
            crate::huffman::encoder::compress(file_in, file_out); 
        }
        ("-huffman", "-d") => { 
            crate::huffman::decoder::decompress(file_in, file_out); 
        }
        ("-bwt", "-c") => { 
            crate::bwt::bwt::bwt_transform(file_in, file_out); 
        }
        ("-bwt", "-d") => {
            // When computing BWT transform, the block size is equal to 
            // the input file buffer size.
            //
            // Because the BWT inverse transform must use the same block 
            // size, this size must be known before creating the BufReader, 
            // but it can't be known before reading it from the file.
            //
            // To get around this, create the file, read first 8 bytes 
            // containing block size, and then wrap it in a BufReader.
            let mut file_in = File::open(file_in_str).unwrap();
            let mut a = [0u8; 8];
            file_in.read(&mut a).unwrap();
            let block_size = u64::from_le_bytes(a) as usize;

            let file_in = BufReader::with_capacity(
                block_size + 8, // Add 8 for primary index size
                file_in
            );
            crate::bwt::bwt::bwt_inverse_transform(file_in, file_out); 
        }
        _ => { 
            print_usage(); 
        }
    }
    
    println!("{} bytes -> {} bytes in {:.2?}", 
        metadata(Path::new(file_in_str)).unwrap().len(), 
        metadata(Path::new(file_out_str)).unwrap().len(), 
        time.elapsed()
    ); 
}

fn print_usage() {
    println!(
        "
        \rUsage: [PROGRAM_NAME] [ALGORITHM] [MODE] [INPUT] [OUTPUT]

        \rALGORITHM:
        \r    -lz77     LZ77 
        \r    -lzw      LZW
        \r    -flzp     LZP
        \r    -fpaq     Adaptive arithmetic encoder
        \r    -lpaq1    Context mixing arithmetic encoder
        \r    -huffman  Static Huffman coding
        \r    -bwt      Burrows-Wheeler transform

        \rMODE:
        \r    -c        Compress
        \r    -d        Decompress

        \rEXAMPLES:
        \r    Compress C:/foo with fpaq and save to C:/bar:

        \r    program_name -fpaq -c C:/foo C:/bar
        "
    );
    std::process::exit(0);
}
