use std::io::BufReader;
use std::io::BufWriter;
use std::fs::File;

use crate::bufio::*;

struct Match {
    pub offset: u16,
    pub len:    u16,
}
impl Match {
    fn new(offset: u16, len: u16) -> Self {
        Self {
            offset,
            len,
        }
    }
}

struct Window {
    data: Vec<u8>,
    pos:  usize,
    size: usize,
}
impl Window {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
            pos:  0,
            size,
        }
    }

    fn add_byte(&mut self, byte: u8) {
        self.data[self.pos % self.size] = byte;
        self.pos += 1;
    }

    fn add_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes.iter() {
            self.add_byte(*byte);
        }
    }

    fn get_byte(&self, pos: usize) -> u8 {
        self.data[pos % self.size]
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

const WINDOW_SIZE: usize = 2048;
const MAX_MATCHES: usize = 512;

pub struct Lz77 {
    window:   Window,
    buf_pos:  usize,
    file_in:  BufReader<File>,
    file_out: BufWriter<File>,
}
impl Lz77 {
    pub fn new(file_in: BufReader<File>, file_out: BufWriter<File>) -> Lz77 {
        Lz77 {
            window:   Window::new(WINDOW_SIZE),
            buf_pos:  0,
            file_in,
            file_out,
        }
    }

    pub fn compress(&mut self) {
        self.file_in.fill_buffer();
        let mut matches = Vec::<Match>::with_capacity(MAX_MATCHES);
        loop {
            for i in (8..self.window.len()).rev() {
                if self.window.get_byte(i) == self.file_in.buffer()[self.buf_pos] {
                    let mut m = Match::new(i as u16, 1);

                    for c in self.file_in.buffer().iter().skip(self.buf_pos + 1).take(30) {
                        if *c == self.window.get_byte((m.offset + m.len) as usize) {
                            m.len += 1;
                        } 
                        else { 
                            break; 
                        }  
                    }
                    if m.len > 1 {
                        matches.push(m);
                    }
                }
                if matches.len() == MAX_MATCHES {
                    break;
                } 
            }
            let best_match = matches.iter().reduce(|best, m| {
                if m.len > best.len { m } else { best }
            });

            if let Some(m) = best_match {
                let ptr = ((m.offset & 0x7FF) << 5) + (m.len & 31);
                self.file_out.write_byte((ptr >> 8) as u8);
                self.file_out.write_byte((ptr & 0x00FF) as u8); 

                let match_bytes = self.buf_pos..self.buf_pos + m.len as usize;
                self.window.add_bytes(&self.file_in.buffer()[match_bytes]); 

                if self.advance(m.len as usize).is_eof() { break; } 
            }
            else {
                self.file_out.write_byte(0);
                self.file_out.write_byte(self.file_in.buffer()[self.buf_pos]);
                self.window.add_byte(self.file_in.buffer()[self.buf_pos]);
                
                if self.advance(1).is_eof() { break; }
            }
            matches.clear();
        } 
        self.file_out.flush_buffer();
    }

    pub fn decompress(&mut self) { 
        self.file_in.fill_buffer(); 
        let mut pending = Vec::new();
        loop {
            let mut ptr = (self.file_in.buffer()[self.buf_pos] as u16) * 256;
            if self.advance(1).is_eof() { break; }
            ptr += self.file_in.buffer()[self.buf_pos] as u16;

            if (ptr >> 8) == 0 {
                self.file_out.write_byte((ptr & 0x00FF) as u8);
                self.window.add_byte(self.file_in.buffer()[self.buf_pos]);
            } 
            else { 
                let m = Match::new(ptr >> 5, ptr & 31);

                for i in 0..m.len {
                    let byte = self.window.get_byte((m.offset + i) as usize);
                    self.file_out.write_byte(byte);
                    pending.push(byte);
                }
                self.window.add_bytes(&pending);
                pending.clear();
            }
            if self.advance(1).is_eof() { break; }
        }
        self.file_out.flush_buffer();
    }

    fn advance(&mut self, len: usize) -> BufferState {
        self.buf_pos += len; 
        if self.buf_pos >= self.file_in.buffer().len() {
            self.buf_pos = 0;
            return self.file_in.fill_buffer()
        }
        BufferState::NotEmpty
    }
}