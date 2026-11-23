//! BC1 blocks from the 12 lanes of a BCPack BC1 stream (one symbol per lane and block).
//!
//! For block `i`, in stream order:
//! * lane 0 is a distance `d` back to the block whose endpoints predict this one's: symbol 0 is
//!   "none" (predict zeros), an even symbol `v` is `v / 2`, an odd one `255 - v / 2`. Block
//!   `i - d` is meant, and before block 0 a built-in history of 255 endpoint sets, the same for
//!   every stream ([`Tables::history`]),
//! * lane 1 was 0 in every stream the format notes describe,
//! * lanes 2–7 are the differences of R0, G0, B0, R1, G1, B1 from the prediction, 0, -1, 1,
//!   -2, 2 … for the symbols 0, 1, 2, 3, 4 …, modulo the channel's range (5, 6, 5 bits),
//! * lanes 8–11 are the selectors of the top-left, top-right, bottom-left and bottom-right 2×2
//!   texels, each symbol the index of one of the 256 possible quadrants in a fixed order
//!   ([`Tables::quadrants`]).
//!
//! The blocks are in the order of the surface's memory, tiled as the console stores it.
//!
//! After the BC1 sections of the public format notes of bcpack-decoder (Tim0219800,
//! `docs/SPEC.md`) and bcpack-rs (Ahmet Dedeler, `docs/BCPACK_SPEC.md`), both MIT; written for
//! XCL on 2026-11-23.

use super::{lanes, lane_widths, Error};

/// The constants of the BC1 rebuild that are not in a stream.
pub struct Tables {
    /// The endpoints `[R0, G0, B0, R1, G1, B1]` of the blocks before a stream's first:
    /// `history[256 - k]` is the block `k` before block 0 (k = 1..=255; entry 0 is never read).
    /// `None`: not known.
    pub history: [Option<[u8; 6]>; 256],
    /// The selectors of each quadrant symbol, 2 bits a texel: bits 0–1 the top-left texel,
    /// 2–3 top-right, 4–5 bottom-left, 6–7 bottom-right.
    pub quadrants: [u8; 256],
}

/// The range of each endpoint channel.
const RANGE: [u8; 6] = [32, 64, 32, 32, 64, 32];

/// Unpack a BC1 stream to its `out_len` bytes of blocks.
pub fn decode(src: &[u8], out_len: usize, tables: &Tables) -> Result<Vec<u8>, Error> {
    let widths = lane_widths(1).unwrap();
    if out_len % 8 != 0 {
        return Err(Error::OutputSize);
    }
    let (symbols, consumed) = lanes::decode(src, widths, out_len / 8)?;
    if src[consumed..].iter().any(|&b| b != 0) {
        return Err(Error::Trailing { consumed });
    }
    assemble(&symbols, tables)
}

/// The endpoints `[R0, G0, B0, R1, G1, B1]` of a block.
fn endpoints(block: &[u8]) -> [u8; 6] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    [(c0 >> 11) as u8, (c0 >> 5 & 63) as u8, (c0 & 31) as u8, (c1 >> 11) as u8, (c1 >> 5 & 63) as u8, (c1 & 31) as u8]
}

/// The blocks of `symbols` (12 per block, in [`super::decode_lanes`]'s order).
pub fn assemble(symbols: &[u8], tables: &Tables) -> Result<Vec<u8>, Error> {
    let blocks = symbols.len() / 12;
    let mut out = Vec::new();
    out.try_reserve_exact(blocks * 8).map_err(|_| Error::OutputSize)?;
    out.resize(blocks * 8, 0u8);
    for (i, s) in symbols.chunks_exact(12).enumerate() {
        if s[1] != 0 {
            return Err(Error::Lane1 { block: i });
        }
        let v = s[0] as usize;
        let d = if v % 2 == 0 { v / 2 } else { 255 - v / 2 };
        let predicted = if d == 0 {
            [0; 6]
        } else if d <= i {
            endpoints(&out[(i - d) * 8..])
        } else {
            let slot = 256 + i - d;
            tables.history[slot].ok_or(Error::History { block: i, slot })?
        };
        let mut e = [0u16; 6];
        for c in 0..6 {
            let v = s[2 + c] as i32;
            let delta = if v % 2 == 0 { v / 2 } else { -(v / 2) - 1 };
            e[c] = (predicted[c] as i32 + delta).rem_euclid(RANGE[c] as i32) as u16;
        }
        let [tl, tr, bl, br] = [8, 9, 10, 11].map(|l| tables.quadrants[s[l] as usize]);
        let b = &mut out[i * 8..i * 8 + 8];
        b[..2].copy_from_slice(&(e[0] << 11 | e[1] << 5 | e[2]).to_le_bytes());
        b[2..4].copy_from_slice(&(e[3] << 11 | e[4] << 5 | e[5]).to_le_bytes());
        // a row of the block is four texels, 2 bits each, the left one lowest
        b[4] = (tl & 15) | (tr & 15) << 4;
        b[5] = (tl >> 4) | (tr >> 4) << 4;
        b[6] = (bl & 15) | (br & 15) << 4;
        b[7] = (bl >> 4) | (br >> 4) << 4;
    }
    Ok(out)
}
