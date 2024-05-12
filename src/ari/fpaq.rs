use std::iter::repeat;
use std::io::BufReader;
use std::io::BufWriter;
use std::fs::File;

use crate::bufio::*;
use crate::ari::log::squash;
use crate::ari::log::stretch;
use crate::ari::state::next_state;

#[allow(overflowing_literals)]
const PR_MSK: i32 = 0xFFFFFE00; // High 23 bit mask
const LIMIT: usize = 127; // Controls rate of adaptation (higher = slower) (0..512)

// StateMap --------------------------------------------------------
struct StateMap {
    cxt:     usize,         
    cxt_map: Vec<u32>,  // Maps a context to a prediction and a count 
    rec:     Vec<u16>,  // Controls adjustment to cxt_map
}

impl StateMap {
    fn new(n: usize) -> Self {
        Self { 
            cxt:     0,
            cxt_map: vec![1 << 31; n],
            rec:     (0..512).map(|i| 32768/(i+i+5)).collect(),
        }
    }

    fn p(&mut self, bit: i32, cxt: usize) -> i32 {
        assert!(bit == 0 || bit == 1);
        self.update(bit);                      
        self.cxt = cxt;
        (self.cxt_map[self.cxt] >> 20) as i32  
    }

    fn update(&mut self, bit: i32) {
        let count = (self.cxt_map[self.cxt] & 511) as usize; // Low 9 bits
        let pr    = (self.cxt_map[self.cxt] >> 14) as i32;   // High 18 bits

        if count < LIMIT {
            self.cxt_map[self.cxt] += 1; 
        }

        let pr_err = (bit << 18) - pr; // Prediction error
        let rec_v = self.rec[count] as i32; // Reciprocal value
        let update = (pr_err * rec_v & PR_MSK) as u32;
        self.cxt_map[self.cxt] = self.cxt_map[self.cxt].wrapping_add(update); 
    }
}

struct Apm {
    bin:  usize,    
    cxts: usize, 
    bins: Vec<u16>,
}

impl Apm {
    fn new(n: usize) -> Self {
        let bins = repeat(
            (0..33)
            .map(|i| (squash((i - 16) * 128) * 16) as u16)
            .collect::<Vec<u16>>()
            .into_iter() 
        )
        .take(n)
        .flatten()
        .collect();

        Self {
            bin:  0,
            cxts: n,
            bins,
        }
    }

    fn p(&mut self, bit: i32, rate: i32, mut pr: i32, cxt: usize) -> i32 {
        assert!(bit == 0 || bit == 1);
        assert!(pr >= 0 && pr < 4096);
        assert!(cxt < self.cxts);

        self.update(bit, rate);
        
        pr = stretch(pr); // -2047 to 2047
        let i_w = pr & 127; // Interpolation weight (33 points)
        
        self.bin = (((pr + 2048) >> 7) + ((cxt as i32) * 33)) as usize;

        let a = self.bins[self.bin] as i32;
        let b = self.bins[self.bin+1] as i32;
        ((a * (128 - i_w)) + (b * i_w)) >> 11
    }

    fn update(&mut self, bit: i32, rate: i32) {
        assert!(bit == 0 || bit == 1);
        assert!(rate > 0 && rate < 32);
        
        // Positive update if bit is 0, negative if 1
        let g = (bit << 16) + (bit << rate) - bit - bit;
        let a = self.bins[self.bin] as i32;
        let b = self.bins[self.bin+1] as i32;
        self.bins[self.bin]   = (a + ((g - a) >> rate)) as u16;
        self.bins[self.bin+1] = (b + ((g - b) >> rate)) as u16;
    }
}

struct Predictor {
    cxt:   usize,         
    cxt4:  usize,        
    pr:    i32,         
    state: [u8; 256],  
    sm:    StateMap, 
    apm:   [Apm; 5],  
}

impl Predictor {
    fn new() -> Self {
        let apm = [
            Apm::new(256),    
            Apm::new(256),   
            Apm::new(65536), 
            Apm::new(8192),
            Apm::new(16384), 
        ];

        Self {
            cxt:   0,                    
            cxt4:  0,                    
            pr:    2048,                 
            state: [0; 256],             
            sm:    StateMap::new(65536), 
            apm,
        }
    }

    fn p(&mut self) -> i32 { 
        assert!(self.pr >= 0 && self.pr < 4096);
        self.pr 
    } 

    fn update(&mut self, bit: i32) {
        assert!(bit == 0 || bit == 1);
        self.state[self.cxt] = next_state(self.state[self.cxt], bit);

        self.cxt += self.cxt + bit as usize;
        if self.cxt >= 256 {
            self.cxt4 = (self.cxt4 << 8) | (self.cxt - 256);
            self.cxt = 0;
        }

        self.pr = self.sm.p(bit, self.state[self.cxt] as usize);

        // SSE
        let cxt = self.cxt;
        self.pr = self.apm[0].p(bit, 5, self.pr, cxt) + 
                  self.apm[1].p(bit, 9, self.pr, cxt) + 1 >> 1;
        
        let cxt = self.cxt | (self.cxt4 << 8) & 0xFF00;
        self.pr = self.apm[2].p(bit, 7, self.pr, cxt);
        
        let cxt = self.cxt | (self.cxt4 & 0x1F00);
        self.pr = self.apm[3].p(bit, 7, self.pr, cxt) * 3 + self.pr + 2 >> 2;

        let hash = (((self.cxt4 as u32) & 0xFFFFFF).wrapping_mul(123456791)) >> 18;
        let cxt = ((self.cxt as u32) ^ hash) as usize;
        self.pr = self.apm[4].p(bit, 7, self.pr, cxt) + self.pr + 1 >> 1;
    }   
}

struct Encoder {
    predictor: Predictor,
    high:      u32,
    low:       u32,
    file_out:  BufWriter<File>,
}

impl Encoder {
    fn new(file_out: BufWriter<File>) -> Self {
        Self {
            predictor: Predictor::new(), 
            high: 0xFFFFFFFF, 
            low: 0,  
            file_out,
        }
    }

    fn encode(&mut self, bit: i32) {
        let p = self.predictor.p() as u32;
        let range = self.high - self.low;
        let mid = self.low + (range >> 12) * p + ((range & 0x0FFF) * p >> 12);

        if bit == 1 { 
            self.high = mid;    
        } 
        else {        
            self.low = mid + 1; 
        }
        self.predictor.update(bit);

        while ((self.high ^ self.low) & 0xFF000000) == 0 {
            self.file_out.write_u8((self.high >> 24) as u8);
            self.high = (self.high << 8) + 255;
            self.low <<= 8;  
        }
    }

    fn flush(&mut self) {
        while ((self.high ^ self.low) & 0xFF000000) == 0 {
            self.file_out.write_u8((self.high >> 24) as u8);
            self.high = (self.high << 8) + 255;
            self.low <<= 8; 
        }
        self.file_out.write_u8((self.high >> 24) as u8);
        self.file_out.flush_buffer();
    }
}

struct Decoder {
    predictor: Predictor,
    high:      u32,
    low:       u32,
    x:         u32,
    file_in:   BufReader<File>,   
}

impl Decoder {
    fn new(file_in: BufReader<File>) -> Self {
        let mut dec = Self {
            predictor: Predictor::new(), 
            high: 0xFFFFFFFF, 
            low: 0, 
            x: 0, 
            file_in, 
        };
        for _ in 0..4 {
            dec.x = (dec.x << 8) + dec.file_in.read_u8() as u32;
        }
        dec
    }

    fn decode(&mut self) -> u8 {
        let p = self.predictor.p() as u32;
        let range = self.high - self.low;
        let mid = self.low + (range >> 12) * p + ((range & 0x0FFF) * p >> 12);

        let mut bit = 0;
        if self.x <= mid {
            bit = 1;
            self.high = mid;
        } 
        else {
            self.low = mid + 1;
        }
        self.predictor.update(bit);
        
        while ((self.high ^ self.low) & 0xFF000000) == 0 {
            self.high = (self.high << 8) + 255;
            self.low <<= 8;
            self.x = (self.x << 8) + self.file_in.read_u8() as u32; 
        }
        bit as u8
    }
}

pub fn fpaq_compress(mut file_in: BufReader<File>, file_out: BufWriter<File>) {
    let mut enc = Encoder::new(file_out);

    while let Some(byte) = file_in.read_u8_checked() { 
        enc.encode(1);
        for i in (0..8).rev() {
            enc.encode(((byte >> i) & 1).into());
        } 
    }   
    enc.encode(0);
    enc.flush(); 
}

pub fn fpaq_decompress(file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let mut dec = Decoder::new(file_in);
            
    while dec.decode() != 0 { 
        let byte = (0..8).fold(1, |acc, _| (acc << 1) + dec.decode());
        file_out.write_u8(byte);
    }
    file_out.flush_buffer();
}
