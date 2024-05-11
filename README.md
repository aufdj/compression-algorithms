# compression-algorithms
A collection of Rust compression algorithms.<br>

## LZ Family Algorithms
* __lz77__: Sliding window compression.
  
* __lzw__: Dictionary compression.
  
* __flzp__[^1]: Byte-oriented LZP compression.

## Arithmetic Encoders
* __fpaq__[^1]: Indirect context modeling arithmetic encoder.
  
* __lpaq1__[^1]: Context mixing arithmetic encoder.

## Other
* __huffman__: Static Huffman coding.
  
* __bwt__: Burrows-Wheeler Transform.



## Usage

        Usage: [PROGRAM_NAME] [ALGORITHM] [MODE] [INPUT] [OUTPUT]

        ALGORITHM:
            -lz77,     LZ77 
            -lzw,      LZW
            -flzp,     LZP
            -fpaq,     Adaptive arithmetic encoder
            -lpaq1,    Context mixing arithmetic encoder
            -huffman,  Static Huffman coding
            -bwt,      Burrows-Wheeler transform

        MODE:
            -c         Compress,
            -d         Decompress,

        EXAMPLES:
            Compress C:/foo with fpaq and save to C:/bar:

            program_name -fpaq -c C:/foo C:/bar


[^1]: Created by [Matt Mahoney](https://mattmahoney.net/dc/dce.html).
