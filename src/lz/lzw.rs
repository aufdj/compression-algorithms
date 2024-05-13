use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::io::BufWriter;

use crate::bufio::*;

const MAX_CODE: u16 = 65535;

pub fn lzw_compress(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let mut dict_code = 256;

    let mut dict = (0..256)
    .map(|i| (vec![i as u8], i))
    .collect::<HashMap<Vec<u8>, u16>>();
    
    let mut string = vec![file_in.read_u8()]; 

    loop {
        while dict.contains_key(&string) {
            if let Some(byte) = file_in.read_u8_checked() {
                string.push(byte); 
            }
            else {
                // EOF reached.
                // Current string is guaranteed to be in dictionary.
                file_out.write_u16(*dict.get(&string).unwrap());
                file_out.flush().unwrap();
                return;
            }  
        }
        dict.insert(string.clone(), dict_code); 
        dict_code += 1;

        let last_char = string.pop().unwrap();
        file_out.write_u16(*dict.get(&string).unwrap());

        string.clear();
        string.push(last_char); 

        if dict_code >= MAX_CODE {
            dict_code = 256;
            dict.retain(|_, i| *i < 256);
        }
    }
}

pub fn lzw_decompress(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let mut dict_code = 256;
    
    let mut dict = (0..256)
    .map(|i| (i, vec![i as u8]))
    .collect::<HashMap<u16, Vec<u8>>>();

    let mut prev_string = Vec::<u8>::with_capacity(64);

    while let Some(code) = file_in.read_u16_checked() {
        if !dict.contains_key(&code) {
            prev_string.push(prev_string[0]);
            dict.insert(code, prev_string);
            dict_code += 1;      
        }
        else if !prev_string.is_empty() {
            prev_string.push((&dict.get(&code).unwrap())[0]);
            dict.insert(dict_code, prev_string);
            dict_code += 1;
        }

        let string = dict.get(&code).unwrap();
        file_out.write(&string).unwrap();

        prev_string = string.to_vec();
        
        if dict_code >= MAX_CODE {
            dict_code = 256;
            dict.retain(|i, _| *i < 256);
        }
    }  
}