//! The entropy stage of BCPack: a set of rANS decoders ("lanes") that share one byte stream.
//!
//! Every lane yields one symbol per 8 bytes of output. A lane's own bytes are a chain of
//! fragments, each a 2- or 3-byte header and its payload:
//! * byte 0: bits 0–3 the model `q`, bit 4 "reset the state when this fragment ends", bit 5
//!   "a third header byte follows", bits 6–7 `z`, by how many bits this fragment's symbols are
//!   narrower than the lane's,
//! * the payload length: byte 1 plus 1, or with a third byte `(byte 2 << 8 | byte 1) + 257`;
//!   it counts every byte the lane reads in the fragment, the first state bytes included.
//!
//! The state is 32 bits with the lower limit `L` = 2^24 and 16-bit probabilities; a byte is
//! shifted in while it is below `L`, a symbol leaves while it is above. A fragment's encoder
//! started at `L`, so a decoded fragment ends there, or at `L >> 8` / `L >> 16` when the
//! encoder dropped trailing zero bytes.
//!
//! Which lane gets which byte of the stream is decided clock by clock, as in the hardware:
//! * each lane holds one byte of input and one symbol of output; in a clock it first applies
//!   its last symbol to the state and takes a byte, then reads a header byte or shifts the byte
//!   in and decodes,
//! * the lanes that ask for a byte are served in lane order, eight bytes a clock at most; a
//!   lane that asked and got none does nothing in that clock,
//! * a lane hands its symbol over only when its place in the symbol vector is free, and the
//!   places are freed together when every lane has filled its own. A lane with a symbol it
//!   cannot hand over does nothing either.
//!
//! The lane is a port of the `decoder` module of Microsoft's silicon-rans sample
//! (`mdu_minirans.v`, Copyright (c) Microsoft Corporation, MIT), made for XCL on 2026-11-23:
//! same registers, same two phases, same control states. The sample leaves the feeder and the
//! symbol vector out; the two rules above are from the public format notes of bcpack-decoder
//! (Tim0219800) and bcpack-rs (Ahmet Dedeler), both MIT, which report them as checked against
//! retail streams.

use super::rans_tables::FREQ;
use super::Error;

/// The lower limit of the rANS state.
pub const L: u32 = 1 << 24;
/// Bytes the feeder hands out per clock.
const FEED_WINDOW: usize = 8;

/// One of the 16 static models at one symbol width: the cumulative base and the per-symbol
/// frequency of each segment. Segment 0 is symbol 0, segment k >= 1 the symbols
/// `2^(k-1) .. 2^k`. Slot 0 belongs to no symbol: symbol 0 starts at slot 1.
#[derive(Clone, Copy)]
struct Model {
    /// `base[k]` for the segments in use, 65536 for the rest
    base: [u32; 10],
    freq: [u32; 9],
}

impl Model {
    fn new(bits: usize, q: usize) -> Self {
        let mut m = Model { base: [65536; 10], freq: [1; 9] };
        let mut at = 0u32;
        for k in 0..=bits {
            m.base[k] = at;
            m.freq[k] = FREQ[bits - 1][q][k] as u32;
            at += m.freq[k] * if k < 2 { 1 } else { 1 << (k - 1) };
        }
        m
    }

    /// The symbol of slot `cf` (1..=65535).
    fn symbol(&self, cf: u32) -> u32 {
        let mut seg = 0;
        for k in 1..9 {
            seg += (cf >= self.base[k]) as usize;
        }
        ((1u32 << seg) >> 1) + (cf - self.base[seg]) / self.freq[seg]
    }

    /// The state after `sym` left state `x`.
    fn advance(&self, x: u32, sym: u32) -> u32 {
        let seg = (32 - sym.leading_zeros()) as usize;
        let cfa = self.base[seg] + self.freq[seg] * (sym - ((1u32 << seg) >> 1));
        let (f, cf) = if cfa == 0 { (self.freq[seg] - 1, 1) } else { (self.freq[seg], cfa) };
        f.wrapping_mul(x >> 16).wrapping_add(x & 0xFFFF).wrapping_sub(cf)
    }
}

/// Where a fragment's decoding may end: the state its encoder started from, less the zero
/// bytes (two at most) the encoder's first symbol pushed out and the encoder did not store.
fn is_end_state(x: u32) -> bool {
    x == L || x == L >> 8 || x == L >> 16
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ctrl {
    Hdr0,
    Hdr1,
    Hdr2,
    Processing,
    /// the fragment's bytes are all in the state; symbols still leave
    Draining,
    Done,
}

struct Lane {
    /// the lane's symbol width
    bits: u8,
    ctrl: Ctrl,
    input_buf: u8,
    input_buf_full: bool,
    sym: u32,
    sym_buf_full: bool,
    /// its place in the symbol vector is taken
    slot_full: bool,
    input_remaining: u32,
    output_remaining: u32,
    /// the state after the last symbol was applied
    p0: u32,
    /// the state the symbol in `sym` was read from
    p1: u32,
    hdr3: bool,
    flush: bool,
    q: u8,
    model: Model,
    stalled: bool,
}

/// Decode the lanes of one stream: `widths[i]` is lane i's symbol width in bits (1..=8),
/// `units` the number of symbols every lane holds. The symbols come back unit by unit
/// (`symbols[unit * lanes + lane]`), with the number of bytes of `src` the lanes read.
pub(super) fn decode(src: &[u8], widths: &[u8], units: usize) -> Result<(Vec<u8>, usize), Error> {
    let mut pos = 0;
    let symbols = run(widths, units, |_| {
        let b = src.get(pos).copied();
        pos += b.is_some() as usize;
        b
    })?;
    Ok((symbols, pos))
}

/// [`decode`] with the stream as a function: the next byte, for the lane that gets it.
pub(super) fn run(widths: &[u8], units: usize, mut next: impl FnMut(usize) -> Option<u8>) -> Result<Vec<u8>, Error> {
    let n = widths.len();
    assert!(n > 0 && widths.iter().all(|w| (1..=8).contains(w)));
    if units == 0 {
        return Ok(Vec::new());
    }
    if units > u32::MAX as usize {
        return Err(Error::OutputSize);
    }
    // grows with the units the stream really yields: the size a request names costs nothing
    let mut symbols: Vec<u8> = Vec::new();
    let mut lanes: Vec<Lane> = widths
        .iter()
        .map(|&bits| Lane {
            bits,
            ctrl: Ctrl::Hdr0,
            input_buf: 0,
            input_buf_full: false,
            sym: 0,
            sym_buf_full: false,
            slot_full: false,
            input_remaining: 0,
            output_remaining: units as u32,
            p0: 0,
            p1: 0,
            hdr3: false,
            flush: false,
            q: 0,
            model: Model::new(bits as usize, 0),
            stalled: false,
        })
        .collect();
    let mut released = 0usize;

    // a lane is done when it has its symbols and has read its last fragment to the end
    while lanes.iter().any(|l| l.ctrl != Ctrl::Done || l.sym_buf_full) {
        let mut progress = false;

        // phase 0: the feeder fills input buffers, a lane applies its symbol and hands it over
        let mut served = 0;
        for (i, l) in lanes.iter_mut().enumerate() {
            let blocked_on_output = l.sym_buf_full && l.slot_full;
            let need = !l.input_buf_full
                && match l.ctrl {
                    Ctrl::Hdr0 | Ctrl::Hdr1 | Ctrl::Hdr2 => true,
                    Ctrl::Processing => l.input_remaining > 0,
                    Ctrl::Draining | Ctrl::Done => false,
                };
            let mut fed = None;
            if need && !blocked_on_output && served < FEED_WINDOW {
                // the stream never grows: a lane that waits for a byte past its end waits forever
                let Some(b) = next(i) else { return Err(Error::Truncated { lane: i }) };
                fed = Some(b);
                served += 1;
            }
            l.stalled = blocked_on_output || (need && fed.is_none());
            if l.stalled {
                continue;
            }
            if matches!(l.ctrl, Ctrl::Processing | Ctrl::Draining) {
                let p0 = if l.sym_buf_full { l.model.advance(l.p1, l.sym) } else { l.p1 };
                progress |= p0 != l.p0;
                l.p0 = p0;
            }
            if let Some(b) = fed {
                l.input_buf = b;
                l.input_buf_full = true;
                progress = true;
            }
            if l.sym_buf_full {
                // the lane's last symbol leaves a state its encoder started from
                let done = units - l.output_remaining as usize;
                if l.output_remaining == 0 {
                    if !is_end_state(l.model.advance(l.p1, l.sym)) {
                        return Err(Error::EndState { lane: i });
                    }
                }
                if symbols.len() < done * n {
                    symbols.try_reserve(n).map_err(|_| Error::OutputSize)?;
                    symbols.resize(done * n, 0);
                }
                symbols[(done - 1) * n + i] = l.sym as u8;
                l.sym_buf_full = false;
                l.slot_full = true;
                progress = true;
            }
        }

        // phase 1: header bytes, or a byte into the state and a symbol out of it
        for (i, l) in lanes.iter_mut().enumerate() {
            if l.stalled {
                continue;
            }
            match l.ctrl {
                Ctrl::Hdr0 => {
                    if l.input_buf_full {
                        let b = l.input_buf;
                        l.input_buf_full = false;
                        l.input_remaining = 0;
                        l.q = b & 15;
                        l.flush = b & 0x10 != 0;
                        l.hdr3 = b & 0x20 != 0;
                        let z = b >> 6;
                        if z >= l.bits {
                            return Err(Error::Width { lane: i });
                        }
                        l.model = Model::new((l.bits - z) as usize, l.q as usize);
                        l.ctrl = Ctrl::Hdr1;
                        progress = true;
                    }
                }
                Ctrl::Hdr1 => {
                    if l.input_buf_full {
                        l.input_buf_full = false;
                        if l.hdr3 {
                            l.input_remaining = l.input_buf as u32;
                            l.ctrl = Ctrl::Hdr2;
                        } else {
                            l.input_remaining = l.input_buf as u32 + 1;
                            l.ctrl = Ctrl::Processing;
                        }
                        progress = true;
                    }
                }
                Ctrl::Hdr2 => {
                    if l.input_buf_full {
                        l.input_buf_full = false;
                        l.input_remaining = ((l.input_buf as u32) << 8 | l.input_remaining) + 257;
                        l.ctrl = Ctrl::Processing;
                        progress = true;
                    }
                }
                Ctrl::Processing => {
                    let feed = l.p0 < L && l.input_remaining > 0 && l.input_buf_full;
                    let x = if feed { l.p0 << 8 | l.input_buf as u32 } else { l.p0 };
                    let out = x > L && l.output_remaining > 0;
                    if out {
                        l.sym = l.model.symbol(x & 0xFFFF);
                        if x & 0xFFFF == 0 {
                            return Err(Error::Symbol { lane: i });
                        }
                        l.output_remaining -= 1;
                    }
                    progress |= feed || out || x != l.p1;
                    l.p1 = x;
                    l.sym_buf_full = out;
                    if feed {
                        l.input_buf_full = false;
                        l.input_remaining -= 1;
                    }
                    if l.input_remaining == 0 {
                        progress = true;
                        if l.output_remaining == 0 {
                            l.ctrl = Ctrl::Done;
                        } else if x == L {
                            l.ctrl = Ctrl::Hdr0;
                            if l.flush {
                                l.p1 = 0;
                            }
                        } else {
                            l.ctrl = Ctrl::Draining;
                        }
                    }
                }
                Ctrl::Draining => {
                    let x = l.p0;
                    let out = x > L && l.output_remaining > 0;
                    if out {
                        l.sym = l.model.symbol(x & 0xFFFF);
                        if x & 0xFFFF == 0 {
                            return Err(Error::Symbol { lane: i });
                        }
                        l.output_remaining -= 1;
                    }
                    progress |= out || x != l.p1;
                    l.p1 = x;
                    l.sym_buf_full = out;
                    if x <= L {
                        progress = true;
                        if l.output_remaining == 0 {
                            l.ctrl = Ctrl::Done;
                        } else {
                            l.ctrl = Ctrl::Hdr0;
                            if l.flush {
                                // The hardware drops whatever state is left here. A state that
                                // is thrown away took symbols with it unless it is where the
                                // fragment's encoder started, so anything else is damage.
                                if !is_end_state(x) {
                                    return Err(Error::EndState { lane: i });
                                }
                                l.p0 = 0;
                                l.p1 = 0;
                            }
                        }
                    }
                }
                Ctrl::Done => {}
            }
        }

        // the symbol vector leaves when every lane has put its symbol in
        if lanes.iter().all(|l| l.slot_full) {
            for l in lanes.iter_mut() {
                l.slot_full = false;
            }
            released += 1;
            progress = true;
        }
        if !progress {
            // nothing moved in a whole clock, so nothing ever will
            return Err(Error::Stuck);
        }
    }
    debug_assert_eq!((released, symbols.len()), (units, units * n));
    Ok(symbols)
}
