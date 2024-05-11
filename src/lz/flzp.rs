/*  flzp file compressor (C) 2008, Matt Mahoney.

    LICENSE

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details at
    Visit <http://www.gnu.org/copyleft/gpl.html>.

To compress:   flzp c input output
To decompress: flzp d input output

flzp is a fast, byte oriented LZP compressor. A byte is decoded
either as a literal, match length, or end of block symbol.
A match length is decoded to the bytes that follow the last
context match within a sliding window.  Uses 8 MB memory.

flzp can be used as a preprocessor to a low-order compressor
to improve both compression ratio and speed.  The following
demonstrates flzp used as a preprocessor to improve compression
for fpaq0 (a stationary 0-order arithmetic coder), and ppmd -o2
and -o3 (PPM orders 2 and 3, flzp hurts compression for higher
orders).  enwik8 is a 100 MB text file.

57,366,279 enwik8.flzp          8 sec
63,391,013 enwik8.fpaq0         36 sec
39,879,072 enwik8.flzp.fpaq0    8+21 sec
36,800,798 enwik8.ppmd-o2
30,884,687 enwik8.flzp.ppmd-o2
30,017,979 enwik8.ppmd-o3
29,372,279 enwik8.flzp.ppmd-o3

A compressed flzp file is decoded as follows. The decoder maintains
the following data structures:

- A rotating buffer BUF[0..BN-1] of length BN bytes, initially all 0.
    An index BUF[i] is understood to mean BUF[i mod BN].
- The number of output bytes, P, initially 0.  The most recent output
    byte is BUF[P - 1].
- A context hash H of the last L bytes output (range 0..BN-1)
    initially 0.
- A hash table HT[0..HN-1] containing HN pointers into the buffer
    (range 0..BN-1), initially all 0.  HT[h] is the largest value
    less than P such that BUF[HT[h]-L..HT[h]-1] has the same hash as
    BUF[P-L..P-1]. Thus, BUF[HT[H]] predicts BUF[P] using an order-L
    context model.
- A decoding table DEC[0..255], where each element is either
    a match length (1..255), LITERAL, or EOB (end of block).
- A parse state STATE, either HEADER or DATA, initially HEADER.

If STATE is HEADER, then the next 32 bytes are used to initialize DEC.
The 256 bits are read in LSB to MSB order.  A 1 bit means LITERAL.
The first 0 bit is stored as EOB.  Subsequent 0 bits are stored
as match lengths in ascending order (1, 2, 3, ..., MAXLEN).  Then
STATE is changed to DATA.

If STATE = DATA then input byte C (0..255) is interpreted as follows:
  If DEC[C] = EOB
    then STATE <- HEADER
  else if DEC[C] = LITERAL then
    output C
    update(C)
  else if DEC[C] = n (match length)
    match <- HT[H]
    For i from 0 to n - 1 do
      c <- BUF[match + i]
      output c
      update(c)

update(c) is defined as
  HT[H] <- P
  H <- update_hash(c)
  BUF[P] <- c
  P <- P + 1 (mod BN)

update_hash(c) is defined as
  H <- (H * M) + c (mod HN)

BN and HN are powers of 2.  M is an odd multiple of ceil(log2(HN)/L).
flzp uses L = 4, BN = 2^22, HN = 2^20, M = 96.  Larger values of BN
and HN generally improve compression at the cost of speed and memory.

Compression requires dividing the input into blocks with 2 passes
over each block. In the first pass, DEC is initialized such that
each byte that appears at least once is initialized to LITERAL
and any remaining entries are initialized to EOB, 1, 2, 3, ..., MAXLEN,
in ascending order, keeping the block small enough to ensure
sufficient match codes.  The block size is as large as possible such
that MAXLEN >= 32 but not larger than 2^16 (65536).

In the second pass, DEC is output as a 32 byte header, then each byte
C of the input is coded by maintaining H, BUF, P, and HT as in decoding.
In addition, the following is maintained:
- The number of pending output bytes, LEN, initially 0 (range 0..MAXLEN).
- The location MATCH in BUF (range 0..BN-1, initially 0) of the
    start of a matching sequence of bytes.

Byte C is compressed:
  If LEN = 0
    then MATCH <- HT[H]
  If C = BUF[MATCH + LEN] and LEN < MAXLEN then
    LEN <- LEN + 1
  else
    output_match()
    MATCH <- HT[H]
    If C = BUF[MATCH] then
      LEN <- 1
    else
      Output C
  update(C)

At the end of a block, any pending matches are output:
  output_match()
  Output EOB

where output_match() is defined
  if LEN > 1 then
    Output n such that DEC[n] = LEN
  else if LEN = 1 then
    Output BUF[P - 1]  (as a literal)
  LEN <- 0

*/

use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Seek;
use std::io::SeekFrom;

use crate::bufio::*;

const BUF_SIZE: usize = 1 << 22;
const HT_SIZE: usize = BUF_SIZE / 4;

struct Buffer {
    buf:     Vec<u8>,    // Rotating buffer of BUF_SIZE bytes
    ht:      Vec<u32>,   // Hash table: hash -> matched context
    enc:     [i32; 256], // Encoding table: -1 = LITERAL, 0 = EOB, 1..max_len = m_pos
    hash:    usize,      // Context hash
    m_pos:   usize,      // Position of match
    m_len:   usize,      // Length of match
    max_len: usize,      // Max length
    p:       usize,      // Number of bytes added to buffer
}

impl Buffer {
    fn new() -> Buffer {
        Buffer {
            buf:     vec![0; BUF_SIZE],
            ht:      vec![0; HT_SIZE],
            enc:     [0; 256],
            hash:    0,
            m_pos:   0,
            m_len:   0,
            max_len: 0,
            p:       0,
        }    
    }

    fn update(&mut self, byte: u8) {
        // Map hash of last L bytes to current buffer position
        self.ht[self.hash] = self.p as u32;      
        // Update hash                                   
        self.hash = (self.hash * 96 + byte as usize) % HT_SIZE; 
        // Update buffer
        self.buf[self.p % BUF_SIZE] = byte; 
        self.p += 1;                                    
    }

    fn update_and_maybe_flush(&mut self, byte: u8, file_out: &mut BufWriter<File>) {
        self.update(byte);   
        // Flush buffer if full                       
        if (self.p % BUF_SIZE) == 0 {  
            file_out.write_all(&self.buf[0..BUF_SIZE]).unwrap();                                    
        }                                           
    }

    fn flush(&mut self, file_out: &mut BufWriter<File>) {
        // Flush remaining bytes
        if (self.p % BUF_SIZE) != 0 {  
            file_out.write_all(&self.buf[0..(self.p % BUF_SIZE)]).unwrap();                                      
        }                      
    }

    fn output_match(&mut self, file_out: &mut BufWriter<File>) {
        if self.m_len > 0 {
            if self.m_len == 1 {
                // Output literal
                file_out.write_byte(self.buf[(self.p - 1) % BUF_SIZE]);
            } 
            else {
                // Output match
                file_out.write_byte(self.enc[self.m_len] as u8);
            }
            self.m_len = 0;
        }
    }

    fn compress(&mut self, byte: u8, file_out: &mut BufWriter<File>) {
        if self.m_len == 0 {
            self.m_pos = self.ht[self.hash] as usize;
        }

        // If subsequent byte matches, increase match length
        let next = (self.m_pos + self.m_len) % BUF_SIZE;
        if self.m_len < self.max_len && self.buf[next] == byte {
            self.m_len += 1;
        } 
        else {
            self.output_match(file_out);
            self.m_pos = self.ht[self.hash] as usize;
            if self.buf[self.m_pos % BUF_SIZE] == byte {
                self.m_len = 1;
            } 
            else {
                file_out.write_byte(byte);
            }
        }
        self.update(byte);
    }
}

pub fn flzp_compress(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let mut buf = Buffer::new();
    
    loop {
        // Pass 1
        let mut dec = [0u8; 32];
        let mut block_size = 0i64;
        buf.max_len = 255;
        
        // Stop if 32 or less unused bytes remain or if block size is greater than 64K.
        while buf.max_len > 32 && block_size < (1 << 16) {
            match file_in.read_byte_checked() {
                Some(byte) => {
                    block_size += 1;
                    // If byte has not been encountered  
                    // before, store in dec.
                    if (dec[byte as usize >> 3] & (1 << (byte & 7))) == 0 {
                        buf.max_len -= 1;
                        dec[byte as usize >> 3] |= 1 << (byte & 7);
                    }
                }
                None => {
                    break;
                }
            }
        }
        if block_size < 1 { 
            break; 
        }

        let mut j = 0usize;
        // Iterate through all bytes and find unused ones.
        for i in 0usize..256 {
            if (dec[i >> 3] & (1 << (i & 7))) == 0 {
                buf.enc[j] = i as i32;
                j += 1;
            }
        }
        assert!(j == (buf.max_len + 1) as usize);

        // Pass 2
        // Seek back to beginning of block
        file_in.seek(SeekFrom::Current(-block_size)).unwrap();

        // Output decoding table as header
        file_out.write_all(&dec[..]).unwrap();

        // Compress
        for _ in 0..block_size {
            buf.compress(file_in.read_byte(), &mut file_out);
        }

        // Output remaining matches
        buf.output_match(&mut file_out);

        // End of block code
        file_out.write_byte(buf.enc[0] as u8);
    }
}

#[derive(PartialEq, Eq)]
enum State {
    Header,
    Data,
} 

pub fn flzp_decompress(mut file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let mut buf = Buffer::new();
    let mut state = State::Header;
    let mut dec = [0i32; 256];

    loop {
        if state == State::Header {
            // Initialize max_len to -1 to store first 0 bit as end of block
            // and subsequent 0 bits as match lengths
            let mut max_len = -1i32;
            for i in 0..32 {
                let byte = file_in.read_byte();
                // Read bits
                for j in 0..8 {
                    dec[i*8 + j] = 
                    if byte & (1 << j) != 0 {
                        // Literal
                        -1 
                    }
                    else {
                        // Match lengths (first is EOB)
                        max_len += 1; 
                        max_len
                    }
                }
            }
            state = State::Data;
        } 
        else {
            match file_in.read_byte_checked() {
                Some(mut byte) => {
                    let d = dec[byte as usize];
                    // End of block
                    if d == 0 { 
                        state = State::Header; 
                    }
                    else if d < 0 {
                        buf.update_and_maybe_flush(byte, &mut file_out);
                    } 
                    else {
                        let mch = buf.ht[buf.hash] as usize;
                        for i in 0..d {
                            byte = buf.buf[(mch + i as usize) % BUF_SIZE];
                            buf.update_and_maybe_flush(byte, &mut file_out);
                        }
                    }
                }
                None => {
                    break;
                }
            }
        }
    }
    buf.flush(&mut file_out);
}
