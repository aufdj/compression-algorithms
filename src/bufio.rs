use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::BufRead;
use std::io::Read;
use std::io::ErrorKind;
use std::convert::TryInto;
use std::mem;

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

fn force_truncate<Src, Dst>(a: Src) -> Dst {
    assert!(mem::size_of::<Src>() > mem::size_of::<Dst>());
    unsafe {
        mem::transmute_copy::<Src, Dst>(&a)
    }
}

// Convenience functions for buffered writing
//
// write_* functions are ideal, but only work with specific types.
//
// write_*_checked functions are for fallible conversions.
//
// write_*_forced is mostly equivalent to 'as', and should only be used 
// in situations where write_* doesn't work but the input is guaranteed to
// fit in the smaller type.
//
// Examples:
//
// let x: u64 = 150;
//
// file.write_byte(x);                  Compile time error: u8 does not implement From<u64>
// file.write_byte_checked(x).unwrap(); No error: x < 255
// file.write_byte_forced(x);           x is interpreted as u8 and byte 150 is written to file
//
// let y: u64 = 500;
//
// file.write_byte(y);                  Compile time error: u8 does not implement From<u64>
// file.write_byte_checked(y).unwrap(); Runtime error: y > 255
// file.write_byte_forced(y);           y is interpreted as u8 and byte 244(!) is written to file
pub trait BufferedWrite {
    fn write_<const N: usize>(&mut self, output: [u8; N]);
    fn write_byte<T: Into<u8>>(&mut self, output: T);
    fn write_u16<T: Into<u16>>(&mut self, output: T);
    fn write_u32<T: Into<u32>>(&mut self, output: T);
    fn write_u64<T: Into<u64>>(&mut self, output: T);
    fn write_byte_checked<T: TryInto<u8>>(&mut self, output: T) -> Result<(), <T as TryInto<u8>>::Error>;
    fn write_u16_checked<T: TryInto<u16>>(&mut self, output: T) -> Result<(), <T as TryInto<u16>>::Error>;
    fn write_u32_checked<T: TryInto<u32>>(&mut self, output: T) -> Result<(), <T as TryInto<u32>>::Error>;
    fn write_u64_checked<T: TryInto<u64>>(&mut self, output: T) -> Result<(), <T as TryInto<u64>>::Error>;
    fn write_byte_forced<T>(&mut self, output: T);
    fn write_u16_forced<T>(&mut self, output: T);
    fn write_u32_forced<T>(&mut self, output: T);
    fn flush_buffer(&mut self);
}

impl BufferedWrite for BufWriter<File> {
    fn write_<const N: usize>(&mut self, output: [u8; N]) {
        self.write(&output[..]).unwrap();
        
        if self.buffer().len() >= self.capacity() {
            self.flush().unwrap();
        }
    }

    fn write_byte<T: Into<u8>>(&mut self, output: T) {
        self.write_(output.into().to_le_bytes());
    }

    fn write_u16<T: Into<u16>>(&mut self, output: T) {
        self.write_(output.into().to_le_bytes());
    }

    fn write_u32<T: Into<u32>>(&mut self, output: T) {
        self.write_(output.into().to_le_bytes());
    }

    fn write_u64<T: Into<u64>>(&mut self, output: T) {
        self.write_(output.into().to_le_bytes());
    }

    fn write_byte_checked<T: TryInto<u8>>(&mut self, output: T) -> Result<(), <T as TryInto<u8>>::Error> {
        self.write_(output.try_into()?.to_le_bytes());
        Ok(())
    }

    fn write_u16_checked<T: TryInto<u16>>(&mut self, output: T) -> Result<(), <T as TryInto<u16>>::Error> {
        self.write_(output.try_into()?.to_le_bytes());
        Ok(())
    }

    fn write_u32_checked<T: TryInto<u32>>(&mut self, output: T) -> Result<(), <T as TryInto<u32>>::Error> {
        self.write_(output.try_into()?.to_le_bytes());
        Ok(())
    }

    fn write_u64_checked<T: TryInto<u64>>(&mut self, output: T) -> Result<(), <T as TryInto<u64>>::Error> {
        self.write_(output.try_into()?.to_le_bytes());
        Ok(())
    }

    fn write_byte_forced<T>(&mut self, output: T) {
        self.write_(force_truncate::<T, u8>(output).to_le_bytes());
    }

    fn write_u16_forced<T>(&mut self, output: T) {
        self.write_(force_truncate::<T, u16>(output).to_le_bytes());
    }

    fn write_u32_forced<T>(&mut self, output: T) {
        self.write_(force_truncate::<T, u32>(output).to_le_bytes());
    }

    fn flush_buffer(&mut self) {
        self.flush().unwrap();
    }
}
