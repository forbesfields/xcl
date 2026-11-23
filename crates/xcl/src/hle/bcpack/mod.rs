//! BCPack, the console's lossless packing of BCn texture blocks (`DSTORAGE_BCPACK_MODE`, the
//! DirectStorage stage between zlib and swizzle): a stream unpacks to plain BCn blocks.
//!
//! A stream has no header. The fields of the blocks are split over 12 or 16 rANS lanes, one
//! symbol per lane for every 8 bytes of output, and the lanes share the byte stream
//! (`lanes.rs`). The blocks are then rebuilt from the symbols; that part differs per format.
//!
//! What is here:
//! * the entropy stage for BC1, BC3, BC4, BC5, BC6H and BC7 ([`lane_widths`]); no lane layout
//!   is published for BC2,
//! * the rebuild of BC1 blocks ([`bc1`]), which needs two tables of constants that are not in
//!   the stream ([`bc1::Tables`]). XCL has the estimates bcpack-rs made of them
//!   (`bc1_fitted.rs`), so its BC1 blocks are an approximation wherever an estimate is off.
//!
//! What is not: the rebuild of the other formats. Their lanes decode, but how the symbols map
//! to block bits is not published in a form XCL can use, so those requests are still refused.
//!
//! Nothing here was compared with what a console writes for the same input: the entropy stage
//! follows Microsoft's published decoder sample and the BC1 rules the public format notes of
//! bcpack-decoder (Tim0219800) and bcpack-rs (Ahmet Dedeler), both MIT, whose authors report
//! them as matching retail streams. The tests check against streams built independently from
//! those notes (`tests/data/bcpack`), not against hardware output.

pub mod bc1;
mod bc1_fitted;
mod lanes;
mod rans_tables;

/// Why a stream was not unpacked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// no block rebuild for this mode (or no lane layout, or no tables)
    Unsupported,
    /// the output is not a whole number of blocks, or too large
    OutputSize,
    /// the stream ends before the lane has its bytes
    Truncated { lane: usize },
    /// a fragment narrows the lane's symbols to nothing
    Width { lane: usize },
    /// the state points at the slot no symbol has
    Symbol { lane: usize },
    /// the lane's last symbol leaves a state no encoder starts from
    EndState { lane: usize },
    /// the lanes wait for each other or for a byte their fragments do not have
    Stuck,
    /// bytes other than zero padding follow the stream
    Trailing { consumed: usize },
    /// BC1 lane 1 is not zero; what it then means is not known
    Lane1 { block: usize },
    /// a block refers to an entry of the built-in history that is not known
    History { block: usize, slot: usize },
}

impl Error {
    /// The kind of failure, without its numbers.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Unsupported => "unsupported",
            Error::OutputSize => "output size",
            Error::Truncated { .. } => "truncated",
            Error::Width { .. } => "width",
            Error::Symbol { .. } => "reserved slot",
            Error::EndState { .. } => "end state",
            Error::Stuck => "stuck",
            Error::Trailing { .. } => "trailing data",
            Error::Lane1 { .. } => "lane 1",
            Error::History { .. } => "unknown history",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Error::Unsupported => write!(f, "no decoder for this mode"),
            Error::OutputSize => write!(f, "the output size is not a whole number of blocks"),
            Error::Truncated { lane } => write!(f, "the stream ends while lane {lane} waits for a byte"),
            Error::Width { lane } => write!(f, "a fragment of lane {lane} narrows its symbols to nothing"),
            Error::Symbol { lane } => write!(f, "lane {lane} reads the reserved slot"),
            Error::EndState { lane } => write!(f, "lane {lane} ends in a state no encoder starts from"),
            Error::Stuck => write!(f, "the lanes cannot go on"),
            Error::Trailing { consumed } => write!(f, "data follows the stream's {consumed} bytes"),
            Error::Lane1 { block } => write!(f, "BC1 block {block} has a non-zero lane 1 symbol"),
            Error::History { block, slot } => write!(f, "BC1 block {block} refers to the unknown history entry {slot}"),
        }
    }
}

/// The symbol width of each lane of a `DSTORAGE_BCPACK_MODE` (1 BC1 … 7 BC7), in stream order.
/// Every lane holds one symbol per 8 bytes of output: one per BC1/BC4 block, two per 16-byte
/// block.
pub fn lane_widths(mode: u8) -> Option<&'static [u8]> {
    Some(match mode {
        1 => &[8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8],
        3 => &[4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8],
        4 => &[8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6],
        5 => &[4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6],
        6 => &[4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5],
        7 => &[4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5],
        _ => return None,
    })
}

/// The symbols of a stream that unpacks to `out_len` bytes, unit by unit
/// (`symbols[unit * lanes + lane]`). Zero padding may follow the stream.
pub fn decode_lanes(mode: u8, src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    let widths = lane_widths(mode).ok_or(Error::Unsupported)?;
    if out_len % 8 != 0 {
        return Err(Error::OutputSize);
    }
    let (symbols, consumed) = lanes::decode(src, widths, out_len / 8)?;
    if src[consumed..].iter().any(|&b| b != 0) {
        return Err(Error::Trailing { consumed });
    }
    Ok(symbols)
}

/// The BC1 constants, when XCL has them: the estimates of bcpack-rs, and in the tests the
/// made-up set their fixtures were built with.
fn bc1_tables() -> Option<&'static bc1::Tables> {
    #[cfg(test)]
    {
        Some(tests::tables())
    }
    #[cfg(not(test))]
    {
        Some(&bc1_fitted::TABLES)
    }
}

/// Whether [`decode`] has a block rebuild for `mode`.
pub fn can_decode(mode: u8) -> bool {
    mode == 1 && bc1_tables().is_some()
}

/// Unpack a stream of `mode` to its `out_len` bytes of BCn blocks.
pub fn decode(mode: u8, src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    match (mode, bc1_tables()) {
        (1, Some(t)) => bc1::decode(src, out_len, t),
        _ => Err(Error::Unsupported),
    }
}

#[cfg(test)]
pub(crate) mod tests;
