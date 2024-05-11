pub const fn squash(d: i32) -> i32 {
    const SQ_T: [i32; 33] = [
    1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,
    1546,2047,2549,2994,3348,3607,3785,3901,3975,4022,
    4050,4068,4079,4085,4089,4092,4093,4094];
    if d > 2047  { return 4095; }
    if d < -2047 { return 0;    }
    let i_w = d & 127;
    let d = ((d >> 7) + 16) as usize;
    (SQ_T[d] * (128 - i_w) + SQ_T[d+1] * i_w + 64) >> 7
}

const STRETCH: [i16; 4096] = build_stretch_table();

const fn build_stretch_table() -> [i16; 4096] {
    let mut table = [0i16; 4096];
    let mut pi = 0;
    let mut x = -2047;
    while x <= 2047 {
        let i = squash(x);
        let mut j = pi;
        while j <= i {
            table[j as usize] = x as i16;
            j += 1;
        }
        pi = i + 1;
        x += 1;
    }
    table[4095] = 2047;
    table
}

pub fn stretch(p: i32) -> i32 {
    STRETCH[p as usize] as i32
}