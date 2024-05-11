use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::BufRead;
use std::io::Read;
use std::io::Result as IoResult;
use std::io::ErrorKind;
use std::path::Path;

#[derive(PartialEq, Eq)]
pub enum BufferState {
    NotEmpty,
    Empty,
}

impl BufferState {
    pub fn is_eof(&self) -> bool {
        *self == BufferState::Empty
    }
}

pub trait BufferedRead {
    fn read_<const N: usize>(&mut self) -> [u8; N];
    fn read_checked<const N: usize>(&mut self) -> Option<[u8; N]>;
    fn read_byte(&mut self) -> u8;
    fn read_u16(&mut self) -> u16;
    fn read_u32(&mut self) -> u32;
    fn read_u64(&mut self) -> u64;
    fn read_byte_checked(&mut self) -> Option<u8>;
    fn read_u16_checked(&mut self) -> Option<u16>;
    fn read_u32_checked(&mut self) -> Option<u32>;
    fn read_u64_checked(&mut self) -> Option<u64>;
    fn fill_buffer(&mut self) -> BufferState;
}

impl BufferedRead for BufReader<File> {
    fn read_<const N: usize>(&mut self) -> [u8; N] {
        let mut bytes = [0u8; N];

        match self.read(&mut bytes) {
            Ok(len) => {
                // If read attempt is partially cut off by end of 
                // buffer, refill buffer and read remaining bytes.
                if self.buffer().is_empty() {
                    self.consume(self.capacity());
                    self.fill_buf().unwrap();
                    if len > 1 && len < N {
                        self.read_exact(&mut bytes).unwrap();
                    }
                }
            }

            Err(e) => {
                panic!("{}", e);
            }
        }
        bytes
    }

    // Read bytes, returning None if not enough bytes could be read.
    // This function is intended for checking if eof has been reached,
    // and will panic upon encountering any errors not caused by reaching eof.
    fn read_checked<const N: usize>(&mut self) -> Option<[u8; N]> {
        let mut bytes = [0u8; N];

        let len = self.read(&mut bytes).unwrap();
        if self.buffer().is_empty() {
            self.consume(self.capacity());
            self.fill_buf().unwrap();
            if len < N {
                match self.read_exact(&mut bytes[len..]) {
                    Ok(_) => {}

                    Err(e) => {
                        if e.kind() == ErrorKind::UnexpectedEof {
                            return None;
                        }
                        else {
                            panic!("{}", e);
                        }
                    }
                }
            }
        }
        Some(bytes)
    }

    fn read_byte(&mut self) -> u8 {
        u8::from_le_bytes(self.read_::<1>())
    }

    fn read_u16(&mut self) -> u16 {
        u16::from_le_bytes(self.read_::<2>())
    }

    fn read_u32(&mut self) -> u32 {
        u32::from_le_bytes(self.read_::<4>())
    }
    
    fn read_u64(&mut self) -> u64 {
        u64::from_le_bytes(self.read_::<8>())
    }

    fn read_byte_checked(&mut self) -> Option<u8> {
        self.read_checked::<1>().map(u8::from_le_bytes)
    }

    fn read_u16_checked(&mut self) -> Option<u16> {
        self.read_checked::<2>().map(u16::from_le_bytes)
    }

    fn read_u32_checked(&mut self) -> Option<u32> {
        self.read_checked::<4>().map(u32::from_le_bytes)
    }

    fn read_u64_checked(&mut self) -> Option<u64> {
        self.read_checked::<8>().map(u64::from_le_bytes)
    }

    fn fill_buffer(&mut self) -> BufferState {
        self.consume(self.capacity());
        self.fill_buf().unwrap();
        if self.buffer().is_empty() {
            return BufferState::Empty;
        }
        BufferState::NotEmpty
    }
}

pub trait BufferedWrite {
    fn write_<const N: usize>(&mut self, output: [u8; N]);
    fn write_byte(&mut self, output: u8);
    fn write_u16(&mut self, output: u16);
    fn write_u32(&mut self, output: u32);
    fn write_u64(&mut self, output: u64);
    fn flush_buffer(&mut self);
}

impl BufferedWrite for BufWriter<File> {
    fn write_<const N: usize>(&mut self, output: [u8; N]) {
        self.write(&output[..]).unwrap();
        
        if self.buffer().len() >= self.capacity() {
            self.flush().unwrap();
        }
    }

    fn write_byte(&mut self, output: u8) {
        self.write_(output.to_le_bytes());
    }

    fn write_u16(&mut self, output: u16) {
        self.write_(output.to_le_bytes());
    }

    fn write_u32(&mut self, output: u32) {
        self.write_(output.to_le_bytes());
    }

    fn write_u64(&mut self, output: u64) {
        self.write_(output.to_le_bytes());
    }

    fn flush_buffer(&mut self) {
        self.flush().unwrap();
    }
}

pub fn new_input_file(capacity: usize, path: &Path) -> IoResult<BufReader<File>> {
    File::open(path).map(|f| BufReader::with_capacity(capacity, f))
}

pub fn new_output_file(capacity: usize, path: &Path) -> IoResult<BufWriter<File>> {
    File::create(path).map(|f| BufWriter::with_capacity(capacity, f))
}