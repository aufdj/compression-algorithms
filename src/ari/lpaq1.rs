use std::iter::repeat;
use std::io::Write;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Seek;
use std::fs::File;
use std::cell::RefCell;
use std::rc::Rc;

use crate::bufio::*;
use crate::ari::log::squash;
use crate::ari::log::stretch;
use crate::ari::state::next_state;
    
const MEM: usize = 1 << 23;

/// An APM takes an existing prediction and a context, and interpolates a 
/// new, refined prediction. Also known as Secondary Symbol Estimation (SSE).
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
            .collect::<Vec<u16>>().into_iter() 
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

/// A bit history (state) is mapped to a probability using an adaptive table
/// (StateMap). Each table entry has a 22-bit probability (initially p = 0.5) 
/// and 10-bit count (initially n = 0) packed into 32 bits.  After bit y is 
/// predicted, n is incremented up to the limit (1023) and the probability is 
/// adjusted by p := p + (y - p)/(n + 0.5).  This model is stationary: 
/// p = (n1 + 0.5)/(n + 1), where n1 is the number of times y = 1 out of n.

#[allow(overflowing_literals)]
const PR_MSK: i32 = 0xFFFFFC00; // High 22 bit mask
const LIMIT: usize = 127; // Controls rate of adaptation (higher = slower) (0..1024)

/// A Statemap is used in an indirect context model to map a context to a 
/// state (a 1 byte representation of previously encountered bits), which 
/// is then mapped to a prediction. 
#[derive(Clone)]
struct StateMap {
    cxt:     usize,    // Context of last prediction
    cxt_map: Vec<u32>, // Maps a context to a prediction and a count
    rec:     Vec<u16>, // Reciprocal table: controls adjustment to cxt_map
}

impl StateMap {
    fn new(n: usize) -> Self {
        Self {
            cxt:     0,
            cxt_map: vec![1 << 31; n],
            rec:     (0..1024).map(|i| 16_384/(i+i+3)).collect(),
        }
    }

    fn p(&mut self, bit: i32, cxt: i32) -> i32 {
        assert!(bit == 0 || bit == 1);
        self.update(bit);
        self.cxt = cxt as usize;
        (self.cxt_map[self.cxt] >> 20) as i32
    }

    /// Update mapping based on prediction error.
    fn update(&mut self, bit: i32) {
        let count = (self.cxt_map[self.cxt] & 1023) as usize; // Low 10 bits
        let pr    = (self.cxt_map[self.cxt] >> 10 ) as i32;   // High 22 bits

        if count < LIMIT { 
            self.cxt_map[self.cxt] += 1; 
        }

        // Update cxt_map based on prediction error
        let pr_err = ((bit << 22) - pr) >> 3; // Prediction error
        let rec_v = self.rec[count] as i32; // Reciprocal value
        let update = ((pr_err * rec_v) & PR_MSK) as u32;
        self.cxt_map[self.cxt] = self.cxt_map[self.cxt].wrapping_add(update);
    }
}


/// Predictions are combined using a neural network (Mixer). The inputs p_i, 
/// i=0..6 are first stretched: t_i = log(p_i/(1 - p_i)), then the output is 
/// computed: p = squash(SUM_i t_i * w_i), where squash(x) = 1/(1 + exp(-x)) 
/// is the inverse of stretch().  The weights are adjusted to reduce the 
/// error: w_i := w_i + L * t_i * (y - p) where (y - p) is the prediction 
/// error and L ~ 0.002 is the learning rate. This is a standard single layer 
/// backpropagation network modified to minimize coding cost rather than RMS 
/// prediction error (thus dropping the factors p * (1 - p) from learning).
struct Mixer {
    max_in:  usize,    // Maximum number of inputs
    inputs:  Vec<i32>, // Current inputs
    weights: Vec<i32>, // Weights used for weighted averaging
    wht_set: usize,    // Single set of weights chosen by a context
    pr:      i32,      // Current prediction
}

impl Mixer {
    /// Create a new Mixer with m sets of n weights.
    fn new(n: usize, m: usize) -> Self {
        Self {
            max_in:   n,                     
            inputs:   Vec::with_capacity(n), 
            weights:  vec![0; n * m],        
            wht_set:  0,                     
            pr:       2048,                  
        }
    }

    /// Add an input prediction to the Mixer.
    fn add(&mut self, pr: i32) {
        assert!(self.inputs.len() < self.inputs.capacity());
        self.inputs.push(pr);
    }

    /// Choose the set of weights to be used for averaging.
    fn set(&mut self, cxt: u32) {
        self.wht_set = (cxt as usize) * self.max_in; 
    }

    /// Compute weighted average of input predictions.
    fn p(&mut self) -> i32 {
        let d = dot_product(&self.inputs[..], &self.weights[self.wht_set..]);
        self.pr = squash(d);
        self.pr
    }

    /// Update weights based on prediction error.
    fn update(&mut self, bit: i32) {
        let error: i32 = ((bit << 12) - self.pr) * 7;
        assert!(error >= -32768 && error < 32768);
        train(&self.inputs[..], &mut self.weights[self.wht_set..], error);
        self.inputs.clear();
    }
}

/// Update weights based on prediction error.
fn train(inputs: &[i32], weights: &mut [i32], error: i32) {
    for (input, weight) in inputs.iter().zip(weights.iter_mut()) {
        *weight += ((*input * error) + 0x8000) >> 16;
    }
}

fn dot_product(inputs: &[i32], weights: &[i32]) -> i32 {
    (inputs.iter().zip(weights.iter())
    .map(|(i, w)| i * w).sum::<i32>()) >> 16
}


/// A match model finds the last occurrence of a high order context and predicts 
/// which symbol came next. The accuracy of the prediction depends on the length 
/// of the context match. The prediction for a match of L bytes (or 8L bits) is 
/// that the next bit will be the same with probability 1 - 1/8L. Typically a 
/// match model of order 6-8 is mixed with lower order context models. A match 
/// model is faster and uses less memory than a corresponding context model but 
/// does not model well for low orders.
///
/// The model looks up the current context in a hash table, first using a longer 
/// context, then a shorter one. If a match is found, then the following bits are 
/// predicted until there is a misprediction. The prediction is computed by mapping 
/// the predicted bit, the match length (1..15 or quantized by 4 in 16..62, max 62),
/// and the last whole byte as context into a StateMap. If no match is found, then 
/// the order 0 context (last 0..7 bits of the current byte) is used as context to 
/// the StateMap.
/// 
/// One third of memory is used by MatchModel, divided equally between a rotating 
/// input buffer of 2^(N+19) bytes and an index (hash table) with 2^(N+17) entries. 
/// Two context hashes are maintained, a long one, h1, of length ceil((N+17)/3) 
/// bytes and a shorter one, h2, of length ceil((N+17)/5) bytes, where ceil() is 
/// the ceiling function. The index does not use collision detection. At each byte 
/// boundary, if there is not currently a match, then the bytes before the current 
/// byte are compared with the location indexed by h1. If less than 2 bytes match, 
/// then h2 is tried. If a match of length >1 is found, the match is maintained 
/// until the next bit mismatches the predicted bit. The table is updated at h1 
/// and h2 after every byte.
const MAX_LEN: usize = 62;

struct MatchModel {
    match_ptr: usize,    // Pointer to current byte in matched context in buf
    match_len: usize,    // Length of match
    cxt:       usize,    // Order-0 context (last 0..7 bits)
    bits:      usize,    // Number of bits in cxt
    hash_s:    usize,    // Short context hash
    hash_l:    usize,    // Long context hash
    buf_pos:   usize,    // Number of bytes in buf
    sm:        StateMap, // Len, bit, last byte -> prediction
    buf:       Vec<u8>,  // Rotating input buffer
    ht:        Vec<u32>, // Context hash -> next byte in buf
    buf_end:   usize,    // Last index of buf (for rotating buffer)
    ht_end:    usize,    // Last index of ht  (for hashing)
}

impl MatchModel {
    fn new(n: usize) -> Self {
        Self {
            match_ptr: 0,    
            match_len: 0,    
            cxt:       1,    
            bits:      0,    
            hash_s:    0,
            hash_l:    0,
            buf_pos:   0,
            sm:        StateMap::new(56 << 8),
            buf:       vec![0; n / 2],
            ht:        vec![0; n / 8],
            buf_end:   (n / 2) - 1,
            ht_end:    (n / 8) - 1,
        }
    }

    /// Generate a prediction and add it to a mixer.
    fn p(&mut self, bit: i32) -> i32 {
        self.update(bit);

        let mut cxt = self.cxt;

        // Get n bits of byte at buf[match_ptr], where n is number of bits in cxt
        // i.e. cxt currently has 3 bits, so get 3 bits of buf[match_ptr]
        let pr_cxt = ((self.buf[self.match_ptr] as usize) + 256) >> (8 - self.bits);

        // If the new value of pr_cxt (containing the next "predicted" bit) doesn't
        // match the new value of cxt (containing the next actual bit), reset the match.
        if self.match_len > 0 && pr_cxt == cxt {
            let pr_bit = (self.buf[self.match_ptr] >> (7 - self.bits) & 1) as usize;

            if self.match_len < 16 { 
                cxt = self.match_len * 2 + pr_bit; 
            }
            else { 
                cxt = (self.match_len >> 2) * 2 + pr_bit + 24; 
            }
            
            let prev = self.buf[(self.buf_pos - 1) & self.buf_end];
            cxt = cxt * 256 + prev as usize;
        } 
        else {
            self.match_len = 0;
        }
        self.sm.p(bit, cxt as i32)
    }

    /// Update context, rotating buffer, and check for matches.
    fn update(&mut self, bit: i32) {
        // Update order-0 context
        self.cxt = (self.cxt << 1) + bit as usize;
        self.bits += 1;                      

        if self.bits == 8 {
            self.update_long_hash(); 
            self.update_short_hash(); 

            // Add byte to buffer
            self.buf[self.buf_pos] = self.cxt as u8; 
            self.buf_pos += 1;            
            self.buf_pos &= self.buf_end; 

            self.bits = 0; 
            self.cxt = 1;

            if self.match_len > 0 { 
                self.match_ptr += 1;
                self.match_ptr &= self.buf_end;
                if self.match_len < MAX_LEN { 
                    self.match_len += 1; 
                }
            }
            // No current match, try long hash
            else { 
                self.find_match(self.hash_l);
            }

            // Less than 2 bytes match, try short hash
            if self.match_len < 2 { 
                self.match_len = 0;
                self.find_match(self.hash_s);
            }

            self.ht[self.hash_s] = self.buf_pos as u32;
            self.ht[self.hash_l] = self.buf_pos as u32;
        }
    }

    /// Check bytes preceding current buffer position and bytes preceding 
    /// buffer position indexed by context hash for matches.
    fn find_match(&mut self, hash: usize) {
        // Map context hash to index in buffer
        self.match_ptr = self.ht[hash] as usize;

        if self.match_ptr != self.buf_pos {
            let mut m1 = (self.match_ptr - self.match_len - 1) & self.buf_end;
            let mut m2 = (self.buf_pos - self.match_len - 1) & self.buf_end;

            // Check subsequent previous bytes, stopping at a mismatch
            while self.match_len < MAX_LEN && m1 != self.buf_pos && self.buf[m2] == self.buf[m1] {
                self.match_len += 1;
                m1 = (m1 - 1) & self.buf_end; 
                m2 = (m2 - 1) & self.buf_end;  
            }
        }
    }
    
    /// Update short hash of length ceil((N+17)/5) bytes.
    fn update_short_hash(&mut self) {
        self.hash_s = (self.hash_s * (5 << 5) + self.cxt) & self.ht_end;
    }

    /// Update long hash of length ceil((N+17)/3) bytes.
    fn update_long_hash(&mut self) {
        self.hash_l = (self.hash_l * (3 << 3) + self.cxt) & self.ht_end;
    }

    /// Return length of match.
    fn len(&self) -> usize {
        self.match_len
    }
}


/// State array length.
const B: usize = 16;

#[repr(align(64))]
#[derive(Debug)]
struct HashTable {
    t:    Vec<u8>, // Hash table mapping index to state array
    size: usize,   // Size of hash table in bytes
}
impl HashTable {
    /// Create a new HashTable.
    fn new(n: usize) -> HashTable {
        assert!(B.is_power_of_two());
        assert!(n.is_power_of_two());
        assert!(n >= (B * 4)); 
        HashTable {
            t:    vec![0; n + B * 4 + 64],
            size: n,
        }
    }

    /// Map context i to element 0 of state array. A state array is a set 
    /// of states corresponding to possible future contexts.
    fn hash(&mut self, mut i: u32) -> *mut u8 {
        i = i.wrapping_mul(123456791).rotate_right(16).wrapping_mul(234567891);
        let chksum = (i >> 24) as u8;
        let mut i = i as usize;
        i = (i * B) & (self.size - B);

        if self.t[i]       == chksum { return &mut self.t[i];       }
        if self.t[i^B]     == chksum { return &mut self.t[i^B];     }
        if self.t[i^(B*2)] == chksum { return &mut self.t[i^(B*2)]; }

        if self.t[i+1] > self.t[(i+1)^B] 
        || self.t[i+1] > self.t[(i+1)^(B*2)] { 
            i ^= B; 
        }

        if self.t[i+1] > self.t[(i+1)^B^(B*2)] { 
            i ^= B ^ (B * 2); 
        }

        self.t[i..i+B].fill(0);
        self.t[i] = chksum;
        &mut self.t[i]
    }
}


type SharedHashTable = Rc<RefCell<HashTable>>;

struct ContextModelO1 {
    bits:      usize,
    pub cxt:   u32,
    pub o1cxt: u32,
    pub state: *mut u8,
    pub t0:    [u8; 65_536],
    sm:        StateMap,
}

impl ContextModelO1 {
    fn new() -> Self {
        Self {
            bits:  0,
            cxt:   1,
            o1cxt: 0,
            state: &mut 0,
            t0:    [0; 65_536], 
            sm:    StateMap::new(256),
        }
    }

    fn p(&mut self, bit: i32) -> i32 {
        self.update(bit);
        unsafe { 
            self.sm.p(bit, *self.state as i32) 
        }
    }

    fn update(&mut self, bit: i32) {
        unsafe { 
            *self.state = next_state(*self.state, bit); 
        }

        self.cxt = (self.cxt << 1) + bit as u32;
        self.bits += 1;

        if self.cxt >= 256 {
            self.cxt -= 256;
            self.o1cxt = self.cxt << 8;
            self.cxt = 1;
            self.bits = 0;
        }

        unsafe { 
            self.state = 
                ((&mut self.t0[0] as *mut u8)
                .add(self.o1cxt as usize))
                .add(self.cxt as usize);
        }
    }
}

struct ContextModelO2 {
    bits:      usize,
    cxt:       u32,
    cxt4:      u32,
    pub o2cxt: u32,
    pub state: *mut u8,
    sm:        StateMap,
    ht:        SharedHashTable,
}

impl ContextModelO2 {
    fn new(ht: SharedHashTable) -> Self {
        Self {
            bits:   0,
            cxt:    1,
            cxt4:   0,
            o2cxt:  0,
            state:  &mut 0,
            sm:     StateMap::new(256),
            ht,
        }
    }

    fn p(&mut self, bit: i32) -> i32 {
        self.update(bit);
        unsafe { 
            self.sm.p(bit, *self.state as i32) 
        }
    }

    fn update(&mut self, bit: i32) {
        unsafe { 
            *self.state = next_state(*self.state, bit); 
        }

        self.cxt = (self.cxt << 1) + bit as u32;
        self.bits += 1;

        if self.cxt >= 256 {
            self.cxt -= 256;
            self.cxt4 = (self.cxt4 << 8) | self.cxt;
            self.o2cxt = (self.cxt4 & 0xFFFF) << 5 | 0x57000000;
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o2cxt).add(1); 
            }
            self.cxt = 1;
            self.bits = 0;
        }
        if self.bits == 4 {
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o2cxt + self.cxt).add(1); 
            }
        }
        else if self.bits > 0 {
            let j = ((bit as usize) + 1) << ((self.bits & 3) - 1);
            unsafe { 
                self.state = self.state.add(j); 
            }
        }
    }
}

struct ContextModelO3 {
    bits:      usize,
    cxt:       u32,
    cxt4:      u32,
    pub o3cxt: u32,
    pub state: *mut u8,
    sm:        StateMap,
    ht:        SharedHashTable,
}

impl ContextModelO3 {
    fn new(ht: SharedHashTable) -> Self {
        Self {
            bits:  0,
            cxt:   1,
            cxt4:  0,
            o3cxt: 0,
            state: &mut 0,
            sm:    StateMap::new(256),
            ht,
        }
    }

    fn p(&mut self, bit: i32) -> i32 {
        self.update(bit);
        unsafe { 
            self.sm.p(bit, *self.state as i32) 
        }
    }

    fn update(&mut self, bit: i32) {
        unsafe { 
            *self.state = next_state(*self.state, bit); 
        }

        self.cxt = (self.cxt << 1) + bit as u32;
        self.bits += 1;

        if self.cxt >= 256 {
            self.cxt -= 256;
            self.cxt4 = (self.cxt4 << 8) | self.cxt;
            self.o3cxt = (self.cxt4 << 8).wrapping_mul(3);
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o3cxt).add(1); 
            }
            self.cxt = 1;
            self.bits = 0;
        }
        if self.bits == 4 {
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o3cxt + self.cxt).add(1); 
            }
        }
        else if self.bits > 0 {
            let j = ((bit as usize) + 1) << ((self.bits & 3) - 1);
            unsafe { 
                self.state = self.state.add(j); 
            }
        }
    }
}

struct ContextModelO4 {
    bits:      usize,
    cxt:       u32,
    cxt4:      u32,
    pub o4cxt: u32,
    pub state: *mut u8,
    sm:        StateMap,
    ht:        SharedHashTable,
}

impl ContextModelO4 {
    fn new(ht: SharedHashTable) -> Self {
        Self {
            bits:  0,
            cxt:   1,
            cxt4:  0,
            o4cxt: 0,
            state: &mut 0,
            sm:    StateMap::new(256),
            ht,
        }
    }

    fn p(&mut self, bit: i32) -> i32 {
        self.update(bit);
        unsafe { 
            self.sm.p(bit, *self.state as i32) 
        }
    }

    fn update(&mut self, bit: i32) {
        unsafe { 
            *self.state = next_state(*self.state, bit); 
        }

        self.cxt = (self.cxt << 1) + bit as u32;
        self.bits += 1;

        if self.cxt >= 256 {
            self.cxt -= 256;
            self.cxt4 = (self.cxt4 << 8) | self.cxt;
            self.o4cxt = self.cxt4.wrapping_mul(5); 
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o4cxt).add(1); 
            }
            self.cxt = 1;
            self.bits = 0;
        }
        if self.bits == 4 {
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o4cxt + self.cxt).add(1); 
            }
        }
        else if self.bits > 0 {
            let j = ((bit as usize) + 1) << ((self.bits & 3) - 1);
            unsafe { 
                self.state = self.state.add(j); 
            }
        }
    }
}

struct ContextModelO6 {
    bits:      usize,
    cxt:       u32,
    cxt4:      u32,
    pub o6cxt: u32,
    pub state: *mut u8,
    sm:        StateMap,
    ht:        SharedHashTable,
}

impl ContextModelO6 {
    fn new(ht: SharedHashTable) -> Self {
        Self {
            bits:  0,
            cxt:   1,
            cxt4:  0,
            o6cxt: 0,
            state: &mut 0,
            sm:    StateMap::new(256),
            ht,
        }
    }

    fn p(&mut self, bit: i32) -> i32 {
        self.update(bit);
        unsafe { 
            self.sm.p(bit, *self.state as i32) 
        }
    }

    fn update(&mut self, bit: i32) {
        unsafe { 
            *self.state = next_state(*self.state, bit); 
        }

        self.cxt = (self.cxt << 1) + bit as u32;
        self.bits += 1;

        if self.cxt >= 256 {
            self.cxt -= 256;
            self.cxt4 = (self.cxt4 << 8) | self.cxt;
            self.o6cxt = (self.o6cxt.wrapping_mul(11 << 5) + self.cxt * 13) & 0x3FFFFFFF;
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o6cxt).add(1); 
            }
            self.cxt = 1;
            self.bits = 0;
        }
        if self.bits == 4 {
            unsafe { 
                self.state = self.ht.borrow_mut().hash(self.o6cxt + self.cxt).add(1); 
            }
        }
        else if self.bits > 0 {
            let j = ((bit as usize) + 1) << ((self.bits & 3) - 1);
            unsafe { 
                self.state = self.state.add(j); 
            }
        }
    }
}

struct WordModel {
    cxt:          u32,
    bits:         usize,
    pub word_cxt: u32,
    pub state:    *mut u8,
    sm:           StateMap,
    ht:           Rc<RefCell<HashTable>>,
}

impl WordModel {
    fn new(ht: Rc<RefCell<HashTable>>) -> Self {
        Self {
            cxt:      1,
            bits:     0,
            word_cxt: 0,
            state:    &mut 0,
            sm:       StateMap::new(256),
            ht,
        }
    }

    fn p(&mut self, bit: i32) -> i32 {
        self.update(bit);
        unsafe { self.sm.p(bit, *self.state as i32) }
    }

    fn update(&mut self, bit: i32) {
        unsafe { *self.state = next_state(*self.state, bit); }

        self.cxt = (self.cxt << 1) + bit as u32;
        self.bits += 1;

        if self.cxt >= 256 {
            self.cxt -= 256;
            self.word_cxt = match self.cxt {
                65..=90 => {
                    self.cxt += 32; // Fold to lowercase
                    (self.word_cxt + self.cxt).wrapping_mul(7 << 3)
                },
                97..=122 => {
                    (self.word_cxt + self.cxt).wrapping_mul(7 << 3)
                },
                _ => 0,
            };
            unsafe { self.state = self.ht.borrow_mut().hash(self.word_cxt).add(1); }
            self.cxt = 1;
            self.bits = 0;
        }
        if self.bits == 4 {
            unsafe { self.state = self.ht.borrow_mut().hash(self.word_cxt + self.cxt).add(1); }
        }
        else if self.bits > 0 {
            let j = ((bit as usize) + 1) << ((self.bits & 3) - 1);
            unsafe { self.state = self.state.add(j); }
        }
    }
}


/// lpaq1 by Matt Mahoney <http://mattmahoney.net/dc/#lpaq>. 
/// lpaq1's model combines 7 contexts: orders 1, 2, 3, 4, 6, a lowercase 
/// unigram word context (for ASCII text), and a "match" order, which 
/// predicts the next bit in the last matching context. The independent 
/// bit predictions of the 7 models are combined using one of 80 neural 
/// networks (selected by a small context), then adjusted using 2 SSE 
/// stages (order 0 and 1) and arithmetic coded.
/// 
/// Prediction is bitwise. This means that an order-n context consists of 
/// the last n whole bytes plus any of the 0 to 7 previously coded bits of 
/// the current byte starting with the most significant bit. The unigram 
/// word context consists of a hash of the last (at most) 11 consecutive 
/// letters (A-Z, a-z) folded to lower case. The context does not include 
/// any nonalphabetic characters nor any characters preceding the last 
/// nonalphabetic character.
/// 
/// The first 6 contexts (orders 1..4, 6, word) are used to index a hash 
/// table to look up a bit-history represented by an 8-bit state. A state 
/// can either represent all histories up to 4 bits long, or a pair of 0,1 
/// counts plus a flag to indicate the most recent bit. The counts are 
/// bounded by (41,0), (40,1), (12,2), (5,3), (4,4) and likewise for 1,0. 
/// If a count is exceeded, the opposite count is reduced to approximately 
/// preserve the count ratio. The last bit flag is present only for states 
/// whose total count is less than 16. There are 253 possible states.
///
/// The 7 predictions are combined using a neural network (Mixer). The
/// inputs p_i, i=0..6 are first stretched: t_i = log(p_i/(1 - p_i)), 
/// then the output is computed: p = squash(SUM_i t_i * w_i), where
/// squash(x) = 1/(1 + exp(-x)) is the inverse of stretch(). The weights
/// are adjusted to reduce the error: w_i := w_i + L * t_i * (y - p) where
/// (y - p) is the prediction error and L ~ 0.002 is the learning rate.
/// This is a standard single layer backpropagation network modified to
/// minimize coding cost rather than RMS prediction error (thus dropping
/// the factors p * (1 - p) from learning).
/// 
/// One of 80 neural networks are selected by a context that depends on
/// the 3 high order bits of the last whole byte plus the context order
/// (quantized to 0, 1, 2, 3, 4, 6, 8, 12, 16, 32). The order is
/// determined by the number of nonzero bit histories and the length of
/// the match from MatchModel.
/// 
/// The Mixer output is adjusted by 2 SSE stages (called APM for adaptive
/// probability map). An APM is a StateMap that accepts both a discrte
/// context and an input probability, pr. pr is stetched and quantized
/// to 24 levels. The output is interpolated between the 2 nearest table 
/// entries, and then only the nearest entry is updated. The entries are 
/// initialized to p = pr and n = 6 (to slow down initial adaptation)
/// with a limit n <= 255. The two stages use a discrete order 0 context 
/// (last 0..7 bits) and a hashed order-1 context (14 bits). Each output 
/// is averaged with its input weighted by 1/4.
/// 
/// The output is arithmetic coded. The code for a string s with probability
/// p(s) is a number between Q and Q+p(x) where Q is the total probability of 
/// all strings lexicographically preceding s. The number is coded as a big-
/// -endian base-256 fraction.
struct Predictor {
    pr:   i32,            // Prediction
    wm:   WordModel,      // Lowercase unigram word model
    mm:   MatchModel,     // Match model
    cm1:  ContextModelO1, // Order 1 context model
    cm2:  ContextModelO2, // Order 2 context model 
    cm3:  ContextModelO3, // Order 3 context model
    cm4:  ContextModelO4, // Order 4 context model
    cm6:  ContextModelO6, // Order 6 context model
    mxr:  Mixer,          // For weighted averaging of independent predictions
    apm1: Apm,            // Adaptive Probability Map for refining Mixer output
    apm2: Apm,            //
}

impl Predictor {
    fn new() -> Predictor {
        // Hash table for mapping context hashes to state arrays.
        // Shared between models.
        let ht = Rc::new(RefCell::new(HashTable::new(MEM*2)));

        let mut p = Predictor {           
            pr:   2048,         
            cm1:  ContextModelO1::new(),
            cm2:  ContextModelO2::new(Rc::clone(&ht)),
            cm3:  ContextModelO3::new(Rc::clone(&ht)),
            cm4:  ContextModelO4::new(Rc::clone(&ht)),
            cm6:  ContextModelO6::new(Rc::clone(&ht)),
            wm:   WordModel::new(Rc::clone(&ht)),
            mm:   MatchModel::new(MEM),
            mxr:  Mixer::new(7, 80),
            apm1: Apm::new(256),
            apm2: Apm::new(16384),
        };
        
        p.wm.state  = &mut p.cm1.t0[0];
        p.cm1.state = &mut p.cm1.t0[0];
        p.cm2.state = &mut p.cm1.t0[0];
        p.cm3.state = &mut p.cm1.t0[0];
        p.cm4.state = &mut p.cm1.t0[0];
        p.cm6.state = &mut p.cm1.t0[0];
        p
    }

    /// Return current prediction.
    fn p(&mut self) -> i32 {
        assert!(self.pr >= 0 && self.pr < 4096);
        self.pr
    }

    /// Update contexts and states, map states to predictions, and mix
    /// predictions in Mixer.
    fn update(&mut self, bit: i32) {
        assert!(bit == 0 || bit == 1);
        
        self.mxr.update(bit);
        
        // Add independent predictions to mixer
        self.mxr.add(stretch(self.mm.p(bit)));
        self.mxr.add(stretch(self.wm.p(bit)));
        self.mxr.add(stretch(self.cm1.p(bit)));
        self.mxr.add(stretch(self.cm2.p(bit)));
        self.mxr.add(stretch(self.cm3.p(bit)));
        self.mxr.add(stretch(self.cm4.p(bit)));
        self.mxr.add(stretch(self.cm6.p(bit)));
        
        // Set weights to be used during mixing
        let order = self.order(self.mm.len());
        self.mxr.set(order + 10 * (self.cm1.o1cxt >> 13));

        // Mix
        self.pr = self.mxr.p();

        // 2 SSE stages
        let cxt = self.cm1.cxt as usize;
        self.pr = (self.pr + 3 * self.apm1.p(bit, 7, self.pr, cxt)) >> 2;

        let cxt = (self.cm1.cxt ^ self.cm1.o1cxt >> 2) as usize;
        self.pr = (self.pr + 3 * self.apm2.p(bit, 7, self.pr, cxt)) >> 2;
    }

    /// Determine order from match model length or number
    /// of non-zero bit histories.
    fn order(&mut self, len: usize) -> u32 {
        let mut order = 0u32;

        // If len is 0, order is determined from 
        // number of non-zero bit histories
        if len == 0 {
            unsafe {
                if *self.cm2.state != 0 { order += 1; }
                if *self.cm3.state != 0 { order += 1; }
                if *self.cm4.state != 0 { order += 1; }
                if *self.cm6.state != 0 { order += 1; }
            }
        }
        else {
            order = 5 +
            if len >= 8  { 1 } else { 0 } +
            if len >= 12 { 1 } else { 0 } +
            if len >= 16 { 1 } else { 0 } +
            if len >= 32 { 1 } else { 0 };
        }
        order
    }
}


struct Encoder {
    high:      u32,
    low:       u32,
    predictor: Predictor,
    archive:   BufWriter<File>,
}

impl Encoder {
    fn new(archive: BufWriter<File>) -> Encoder {
        let mut enc = Encoder {
            high: 0xFFFFFFFF, 
            low: 0, 
            predictor: Predictor::new(), 
            archive
        };   
        enc.archive.write_u64(0u64);
        enc.archive.write_u64(0u64);
        enc.archive.write_u64(0u64);
        enc
    }

    fn encode_bit(&mut self, bit: i32) {
        let mut p = self.predictor.p() as u32;
        if p < 2048 { p += 1; }

        let range = self.high - self.low;
        let mid: u32 = self.low + (range >> 12) * p
                       + ((range & 0x0FFF) * p >> 12);
                       
        if bit == 1 {
            self.high = mid;
        } 
        else {
            self.low = mid + 1;
        }
        self.predictor.update(bit);
        
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            self.archive.write_u8_forced(self.high >> 24);
            self.high = (self.high << 8) + 255;
            self.low <<= 8;  
        }
    }

    fn flush(&mut self) {
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            self.archive.write_u8_forced(self.high >> 24);
            self.high = (self.high << 8) + 255;
            self.low <<= 8;
        }
        self.archive.write_u8_forced(self.high >> 24);
        self.archive.flush_buffer();
    }

    fn encode_block(&mut self, block: &[u8]) {
        for byte in block.iter() {
            for i in (0..=7).rev() {
                self.encode_bit(((*byte >> i) & 1) as i32);
            }   
        }
    }

    // Write 24 byte block data header
    fn write_block_data(&mut self, data: BlockData) {
        self.archive.get_ref().rewind().unwrap();
        self.archive.write_u64(data.final_size);
        self.archive.write_u64(data.base_size);
        self.archive.write_u64(data.count);    
    }
}


struct Decoder {
    high:      u32,
    low:       u32,
    predictor: Predictor,
    archive:   BufReader<File>,
    x:         u32, 
}

impl Decoder {
    fn new(archive: BufReader<File>) -> Self {
        Self {
            high: 0xFFFFFFFF, 
            low: 0, 
            x: 0, 
            predictor: Predictor::new(), 
            archive,
        }
    }

    fn decode_bit(&mut self) -> i32 {
        let mut p = self.predictor.p() as u32;
        if p < 2048 { p += 1; }

        let range = self.high - self.low;
        let mid = self.low + (range >> 12) * p + ((range & 0x0FFF) * p >> 12);

        let mut bit: i32 = 0;
        if self.x <= mid {
            bit = 1;
            self.high = mid;
        } 
        else {
            self.low = mid + 1;
        }
        self.predictor.update(bit);
        
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            self.high = (self.high << 8) + 255;
            self.low <<= 8; 
            self.x = (self.x << 8) + self.archive.read_u8() as u32; 
        }
        bit
    }

    fn decode_block(&mut self, block_size: u64) -> Vec<u8> {
        let mut block = Vec::with_capacity(block_size as usize); 
        while block.len() < block.capacity() {
            let mut byte = 1;
            while byte < 256 {
                byte += byte + self.decode_bit();
            }
            byte -= 256;
            block.push(byte as u8); 
        }
        block
    }

    // Read 24 byte block data header
    fn read_block_data(&mut self) -> BlockData {
        BlockData::from(
            self.archive.read_u64(),
            self.archive.read_u64(),
            self.archive.read_u64()
        )
    }

    fn init_x(&mut self) {
        for _ in 0..4 {
            self.x = (self.x << 8) + self.archive.read_u8() as u32;
        }
    }
}


struct BlockData {
    base_size:  u64,
    final_size: u64,
    count:      u64,
}
impl BlockData {
    fn new(base_size: u64) -> Self {
        Self {
            base_size,
            final_size: 0,
            count: 0,
        }
    }

    fn from(final_size: u64, base_size: u64, count: u64) -> Self {
        Self {
            base_size,
            final_size,
            count
        }
    }

    fn update(&mut self, size: u64) {
        self.final_size = size;
        self.count += 1;
    }
}

pub fn lpaq1_compress(mut file_in: BufReader<File>, file_out: BufWriter<File>) {
    let mut data = BlockData::new(file_in.capacity() as u64);
    let mut enc = Encoder::new(file_out);

    while !file_in.fill_buffer().is_eof() {
        data.update(file_in.buffer().len() as u64);
        enc.encode_block(&file_in.buffer());
    } 
    enc.flush();
    enc.write_block_data(data);
}

pub fn lpaq1_decompress(file_in: BufReader<File>, mut file_out: BufWriter<File>) {
    let mut dec = Decoder::new(file_in);
    let data = dec.read_block_data();

    // Call after reading header
    dec.init_x();

    for _ in 0..(data.count - 1) {
        file_out.write_all(&dec.decode_block(data.base_size)).unwrap();
    }
    file_out.write_all(&dec.decode_block(data.final_size)).unwrap();
    file_out.flush_buffer();
}
