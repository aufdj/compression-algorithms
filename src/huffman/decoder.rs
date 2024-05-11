use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::BufRead;
use std::fs::File;
use std::array;

use crate::bufio::*;

use crate::huffman::huffman::Node;
use crate::huffman::huffman::NodeType;

pub fn decompress(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let file_in_size = file_in.get_ref().metadata().unwrap().len();
    let padding = file_in.read_byte();

    let frequencies: [u32; 256] = array::from_fn(|_| file_in.read_u32());

    let mut heap = BinaryHeap::with_capacity(512);
    for (i, frequency) in frequencies.iter().enumerate() {                                               
        heap.push(                                                  
            Node::new(
                *frequency,
                NodeType::Leaf(i as u8)
            )
        );
    }   

    build_tree(&mut heap); 

    let mut codes = HuffmanCodeMap::new();
    gen_codes(heap.peek().unwrap(), vec![], &mut codes);

    let mut curr_code: Vec<u8> = Vec::with_capacity(8);
    let mut pos = 1026;
    file_in.fill_buf().unwrap();
    
    loop {
        for byte in file_in.buffer().iter() {
            if pos >= file_in_size {
                for j in (0..=(7 - padding)).rev() {
                    curr_code.push((*byte >> j) & 1);
                    if let Some(byte) = codes.get(&curr_code) {
                        file_out.write_byte(*byte);
                        curr_code.clear();
                    }
                }
            } 
            else {
                for j in (0..=7).rev() {
                    curr_code.push((*byte >> j) & 1);
                    if let Some(byte) = codes.get(&curr_code) {
                        file_out.write_byte(*byte);
                        curr_code.clear();
                    }
                }
            }
            pos += 1;
        }
        if file_in.fill_buffer() == BufferState::Empty {
            file_out.flush_buffer();
            break;
        }
    }   
}

type HuffmanCodeMap = HashMap<Vec<u8>, u8>;

fn gen_codes(node: &Node, prefix: Vec<u8>, codes: &mut HuffmanCodeMap) {
    match node.node_type {
        NodeType::Internal(ref left_child, ref right_child) => {
            let mut left_prefix = prefix.clone();
            left_prefix.push(0);
            gen_codes(left_child, left_prefix, codes);

            let mut right_prefix = prefix;
            right_prefix.push(1);
            gen_codes(right_child, right_prefix, codes);
        }
        NodeType::Leaf(byte) => {
            codes.insert(prefix, byte);
        }
    }
}

fn build_tree(heap: &mut BinaryHeap<Node>) {
    while heap.len() > 1 {
        let left_child = heap.pop().unwrap();
        let right_child = heap.pop().unwrap();
        heap.push(
            Node::new(
                left_child.frequency + right_child.frequency, 
                NodeType::Internal(
                    Box::new(left_child), 
                    Box::new(right_child)
                )
            )
        );
    }
}