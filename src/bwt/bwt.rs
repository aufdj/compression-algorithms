use std::io::Write;
use std::io::BufReader;
use std::io::BufWriter;
use std::fs::File;
use std::cmp::Ordering;
use std::cmp::min;

use crate::bufio::*;

pub fn bwt_transform(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    file_out.write_u64(file_in.capacity() as u64);

    loop {
        if file_in.fill_buffer().is_eof() { 
            break; 
        }

        let len = file_in.buffer().len();

        let mut indices = (0..len as u32).collect::<Vec<u32>>();

        indices.sort_by(|a, b| {
            block_cmp(*a as usize, *b as usize, file_in.buffer())
        });

        let mut primary_index = None;

        let bwt = indices.iter().enumerate().map(|(i, &idx)| {
            match idx {
                0 => {
                    file_in.buffer()[len - 1]
                }

                1 => {
                    primary_index = Some(i);
                    file_in.buffer()[idx as usize - 1]
                }

                _ => {
                    file_in.buffer()[idx as usize - 1]
                }
            }  
        })
        .collect::<Vec<u8>>();
    
        file_out.write_u64(primary_index.unwrap() as u64);
        file_out.write_all(&bwt).unwrap();
    }  
    file_out.flush_buffer();
}

pub fn bwt_inverse_transform(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let mut transform = vec![0u32; file_in.capacity()];

    loop {
        if file_in.fill_buffer().is_eof() { 
            break; 
        }
    
        let mut index = file_in.read_u64() as usize;

        let mut count = [0u32; 256];
        let mut cumul = [0u32; 256];

        for byte in file_in.buffer().iter() {
            count[*byte as usize] += 1;    
        }

        let mut sum = 0;
        for i in 0..256 {
            cumul[i] = sum;
            sum += count[i];
            count[i] = 0;
        }

        for (i, byte) in file_in.buffer().iter().enumerate() {
            let byte = *byte as usize;
            transform[(count[byte] + cumul[byte]) as usize] = i as u32;
            count[byte] += 1;
        }

        for _ in 0..file_in.buffer().len() { 
            file_out.write_byte(file_in.buffer()[index]);
            index = transform[index] as usize;
        }
    }
    file_out.flush().unwrap();
}

fn block_cmp(a: usize, b: usize, block: &[u8]) -> Ordering {
    let min = min(block[a..].len(), block[b..].len());

    // Lexicographical comparison
    let result = block[a..a + min].cmp(&block[b..b + min]);
    
    // Wraparound if needed
    if result == Ordering::Equal {
        let remainder_a = [&block[a + min..], &block[0..a]].concat();
        let remainder_b = [&block[b + min..], &block[0..b]].concat();
        return remainder_a.cmp(&remainder_b);
    }
    result   
}
