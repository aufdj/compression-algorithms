use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Seek;
use std::fs::File;

use crate::bufio::*;

use crate::huffman::huffman::Node;
use crate::huffman::huffman::NodeType;

pub fn compress(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    file_out.write_u8(0);

    // Model data to get frequency distribution
    let frequencies: [u32; 256] = model(&mut file_in);
     
    // Include model as compressed data header
    for freq in frequencies.iter() {                                
        file_out.write_u32(*freq);                                                                                               
    }                                                               
    
    // Add leaf nodes to heap
    let mut heap: BinaryHeap<Node> = BinaryHeap::new();
    for i in 0..256 {                                               
        heap.push(
            Node::new(
                frequencies[i], 
                NodeType::Leaf(i as u8)
            )      
        );                                                          
    }                                                               

    build_tree(&mut heap);       

    // Walk down tree and generate codes
    let mut codes = HuffmanCodeMap::new();                          
    gen_codes(heap.peek().unwrap(), vec![], &mut codes);

    file_in.rewind().unwrap();

    let mut packed_codes: u8 = 0;
    let mut bits: u8 = 0;

    while file_in.fill_buffer() == BufferState::NotEmpty {
        for byte in file_in.buffer().iter() { 
            // Get huffman code corresponding to current byte and write bits to output
            for bit in codes.get(byte).unwrap() {
                if bits >= 8 {
                    file_out.write_u8(packed_codes);
                    packed_codes = 0;
                    bits = 0;
                }
                packed_codes = (packed_codes << 1) + bit;
                bits += 1;
            }
        }
    } 
    // Write remaining code
    if bits > 0 {
        file_out.write_u8(packed_codes);
    }
    file_out.flush_buffer();
    file_out.rewind().unwrap();

    // Write number of padding bits
    file_out.write_u8(8 - bits);
}

// Model data to get frequency distribution
fn model(file_in: &mut BufReader<File>) -> [u32; 256] {
    let mut frequencies = [1u32; 256];
    while file_in.fill_buffer() == BufferState::NotEmpty {
        for byte in file_in.buffer().iter() {
            frequencies[*byte as usize] += 1;
        }
    }
    frequencies
}

// Build tree from leaf nodes
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


// Walk down every branch of tree to get codes for every byte
type HuffmanCodeMap = HashMap<u8, Vec<u8>>;

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
            codes.insert(byte, prefix);
        }
    }
}