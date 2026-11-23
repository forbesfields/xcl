//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).
//!
//! Textures the CPU fills (kept in step with the title's memory, `title_textures`) go on pages
//! the same way, in a space of their own (`Space`): a title streaming its world places new
//! textures over memory it used before and lets few of the old ones go, and a committed texture
//! per placement held the sum of every texture ever placed (#248).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    heap: H,
    /// resources placed in it
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Arenas no resource used for `IDLE` go (their heap with the last reference).
    fn sweep(&mut self, now: std::time::Instant) {
        self.list.retain(|a| a.users > 0 || a.idle_since.is_none_or(|t| now.duration_since(t) < IDLE));
    }

    /// Where a placement of `bytes` at `va` (alignment `align`) goes: the newest arena holding
    /// it at an offset `align` allows, else a new arena of its size, `make(size, align)`.
    /// (heap, offset, arena id); None: no heap.
    fn take(&mut self, va: usize, bytes: usize, align: usize, now: std::time::Instant, make: impl FnOnce(usize, usize) -> Option<H>) -> Option<(H, usize, u64)> {
        self.sweep(now);
        let fits = |a: &Arena<H>| a.align >= align && a.base <= va && va + bytes <= a.base + a.size && (va - a.base) % align == 0;
        let i = match self.list.iter().rposition(fits) {
            Some(i) => i,
            None => {
                let size = bytes.div_ceil(align) * align;
                let heap = make(size, align)?;
                self.next_id += 1;
                self.list.push(Arena { base: va, size, align, heap, users: 0, idle_since: None, id: self.next_id });
                self.list.len() - 1
            }
        };
        let a = &mut self.list[i];
        a.users += 1;
        a.idle_since = None;
        Some((a.heap.clone(), va - a.base, a.id))
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            a.users = a.users.saturating_sub(1);
            if a.users == 0 {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
/// resource → its arena's id
static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource));
    if placed {
        forget_pages(resource);
        if let Some(id) = ARENA_OF.lock().unwrap().as_mut().and_then(|m| m.remove(&resource)) {
            ARENAS.lock().unwrap().release(id, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
        return None;
    }
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    heap
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

/// Which resources share a title page's video memory. GPU-owned targets share with targets,
/// textures kept in step with the title's memory (`title_textures`) with such textures: their
/// uploads are the CPU's view of the memory, which over a target's page would overwrite what
/// the GPU rendered there (on the console the memory holds it; here the CPU's view does not).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Space {
    Targets,
    Textures,
    TextureTail(usize),
}

struct Page {
    tile: Tile,
    /// resources using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
}

/// Which pool tile backs which title page, and which resource uses which pages – apart from
/// D3D12 (a new pool heap is the `grow` callback's).
struct Book {
    free: Vec<Tile>,
    /// tiles given back, reusable after `COOL`
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
    /// Candidates for retirement in release order. A new placement can reactivate a page.
    idle: std::collections::VecDeque<((Space, usize), std::time::Instant)>,
    /// (space, title page address) → its video memory
    pages: std::collections::HashMap<(Space, usize), Page>,
    /// resource → (shared page keys, tiles of its own past the title's memory)
    users: std::collections::HashMap<usize, (Vec<(Space, usize)>, Vec<Tile>)>,
}

impl Book {
    fn new() -> Book {
        Book { free: Vec::new(), cooling: std::collections::VecDeque::new(), idle: std::collections::VecDeque::new(), pages: std::collections::HashMap::new(), users: std::collections::HashMap::new() }
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        while let Some(&(key, released)) = self.idle.front() {
            if now.duration_since(released) < IDLE {
                break;
            }
            self.idle.pop_front();
            if self.pages.get(&key).is_some_and(|page| page.idle_since == Some(released)) {
                if let Some(page) = self.pages.remove(&key) {
                    self.cooling.push_back((page.tile, now));
                }
            }
        }
    }

    /// A tile: a free one, one cooled down, else the first of a new pool heap (`grow`: its
    /// index; None: no video memory).
    fn take_tile(&mut self, now: std::time::Instant, grow: &mut impl FnMut() -> Option<u32>) -> Option<Tile> {
        if let Some(t) = self.free.pop() {
            return Some(t);
        }
        if self.cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL) {
            return self.cooling.pop_front().map(|(t, _)| t);
        }
        let h = grow()?;
        self.free.extend((1..POOL_PAGES).rev().map(|t| (h, t)));
        Some((h, 0))
    }

    /// The tiles of resource `res` at title address `va` in `space`: `total` tiles, the first
    /// `own` on the title's pages there (shared with every resource of the space on them), the
    /// rest its own. None: no video memory for all of them (nothing is kept).
    #[allow(clippy::too_many_arguments)]
    fn claim(&mut self, space: Space, res: usize, va: usize, own: usize, total: usize, now: std::time::Instant, grow: &mut impl FnMut() -> Option<u32>) -> Option<Vec<Tile>> {
        self.sweep(now);
        let first = va.next_multiple_of(PAGE);
        let mut tiles = Vec::with_capacity(total);
        let mut used = Vec::with_capacity(own.min(total));
        let mut extra = Vec::new();
        for i in 0..total {
            let key = if i < own {
                Some((space, first + i * PAGE))
            } else if space == Space::Textures && va != first {
                // Small textures with different starting addresses can share a console page.
                // Give their padding separate tiles, shared only across repeated placements.
                Some((Space::TextureTail(va), i))
            } else {
                None
            };
            let tile = if let Some(key) = key {
                let tile = match self.pages.get_mut(&key) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        Some(p.tile)
                    }
                    None => self.take_tile(now, grow).inspect(|&t| {
                        self.pages.insert(key, Page { tile: t, users: 1, idle_since: None });
                    }),
                };
                if tile.is_some() {
                    used.push(key);
                }
                tile
            } else {
                self.take_tile(now, grow).inspect(|&t| extra.push(t))
            };
            let Some(t) = tile else { break };
            tiles.push(t);
        }
        self.users.insert(res, (used, extra));
        if tiles.len() < total {
            self.release(res, now);
            return None;
        }
        Some(tiles)
    }

    /// Resource `res` released: its pages are no longer its.
    fn release(&mut self, res: usize, now: std::time::Instant) {
        let Some((used, extra)) = self.users.remove(&res) else { return };
        for key in used {
            if let Some(p) = self.pages.get_mut(&key) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    p.idle_since = Some(now);
                    self.idle.push_back((key, now));
                }
            }
        }
        self.cooling.extend(extra.into_iter().map(|t| (t, now)));
    }

    /// bytes of the pages (idle ones included) and own tiles in use
    fn bytes(&self) -> usize {
        (self.pages.len() + self.users.values().map(|u| u.1.len()).sum::<usize>()) * PAGE
    }
}

/// The title's pages of video memory for textures in its memory (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    pools: Vec<ID3D12Heap>,
    book: Book,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// None inside: no pages on this device (not tried yet: the outer None)
static PAGES: Mutex<Option<Option<Pages>>> = Mutex::new(None);

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages { queue, fence, value: 0, pools: Vec::new(), book: Book::new(), waited: std::collections::HashMap::new() })
    }

    /// A new pool heap (its index); None: no video memory.
    unsafe fn grow(dev: &ID3D12Device, pools: &mut Vec<ID3D12Heap>) -> Option<u32> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            // every kind of texture (resource heap tier 2: the only flags for both)
            Flags: D3D12_HEAP_FLAG_NONE,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
            crate::xlog_warn!("D3D12.X: no video memory for more pages of textures in title memory ({e})");
            return None;
        }
        pools.push(heap?);
        Some(pools.len() as u32 - 1)
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i) – one call per pool heap.
    unsafe fn map(&mut self, res: &ID3D12Resource, tiles: &[Tile]) {
        let mut heaps: Vec<u32> = tiles.iter().map(|t| t.0).collect();
        heaps.sort_unstable();
        heaps.dedup();
        for h in heaps {
            let ranges = heap_ranges(h, tiles);
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|r| if r.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
            let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
            let counts: Vec<u32> = ranges.iter().map(|r| r.2).collect();
            let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
            let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let mine = h == heap;
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// Number of 64 KiB-aligned addresses inside the texture's console allocation.
fn covered_pages(va: usize, bytes: usize) -> usize {
    let Some(first) = va.checked_next_multiple_of(PAGE) else { return 0 };
    va.saturating_add(bytes).saturating_sub(first).div_ceil(PAGE)
}

/// A texture at `va` as a reserved resource on the title's pages there in `space`
/// (`console_bytes`: its size on the console – the pages it covers; tiles past them are its
/// own). None: it cannot be (the caller tries an arena or a resource of its own).
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(dev: &ID3D12Device, this: *mut c_void, space: Space, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    static UNALIGNED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let unaligned = va % PAGE != 0;
    if unaligned && (space != Space::Textures || va % 4096 != 0 || !*UNALIGNED.get_or_init(|| std::env::var("XCL_UNALIGNED_TEXTURE_PAGES").is_ok_and(|v| v.trim() == "1"))) {
        return None;
    }
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    let mut g = PAGES.lock().unwrap();
    let pages = g.get_or_insert_with(|| Pages::new(dev)).as_mut()?;
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: texture at {va:#x} ({space:?}: {}x{}, format {}, {} samples) not reserved ({hr:#x}): not on pages", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
        }
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
    if total == 0 {
        drop(g);
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    }
    let own = console_bytes.map_or(total as usize, |b| covered_pages(va, b as usize)).min(total as usize);
    let Pages { book, pools, .. } = pages;
    let Some(tiles) = book.claim(space, raw as usize, va, own, total as usize, std::time::Instant::now(), &mut || Pages::grow(dev, pools)) else {
        // no video memory for all of it: the texture goes
        drop(g);
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    };
    pages.map(res, &tiles);
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    Some(hr)
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.as_mut() else { return };
    pages.book.release(resource, std::time::Instant::now());
}

/// A texture the title placed at `va` whose texels the CPU writes (kept in step with the memory,
/// `title_textures`) as a reserved resource on the title's pages of such textures (`desc`: the PC
/// description, `console_bytes` its size on the console). A title streams its world by placing
/// new textures over memory it used before – as the console allows: the new texture overwrites
/// the old one's memory – and lets few of the old ones go; a committed texture per placement
/// held the sum of every texture ever placed (a retail game: 12.7 GB of them over 4.0 GB of its
/// memory after 90 s, then the device was removed). On pages they cost the memory they cover.
/// None: not on pages (the caller makes a resource of its own). `XCL_TITLE_TEXTURE_PAGES=0`
/// turns this off.
pub unsafe fn place_title_texture(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, console_bytes: u64, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("XCL_TITLE_TEXTURE_PAGES").is_ok_and(|v| v.trim() == "0")) {
        return None;
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    place_on_pages(dev, this, Space::Textures, va, desc, state, std::ptr::null(), Some(console_bytes), riid, ppv)
}

/// Before a title queue's work: it waits for the pages mapped since its last work.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.as_mut() else { return };
    let key = queue.as_raw() as usize;
    if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
        let _ = queue.Wait(&pages.fence, pages.value);
        pages.waited.insert(key, pages.value);
    }
}

/// (bytes of pages in use by targets and textures, bytes of pool heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.as_ref() else { return (0, 0) };
    (p.book.bytes(), p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let Some(hr) = place_on_pages(dev, this, Space::Targets, va, desc, state, clear, console_bytes, riid, ppv) {
            return Some(hr);
        }
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let align = (info.Alignment as usize).max(PAGE);
    let now = std::time::Instant::now();
    let (heap, offset, id) = ARENAS.lock().unwrap().take(va, info.SizeInBytes as usize, align, now, |size, align| arena_heap(dev, va, size, align))?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        ARENAS.lock().unwrap().release(id, now);
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    ARENA_OF.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(*ppv as usize, id);
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    /// A pool-heap maker for `Book` that numbers heaps as they are made.
    fn heaps() -> impl FnMut() -> Option<u32> {
        let mut n = 0;
        move || {
            n += 1;
            Some(n - 1)
        }
    }

    /// Textures at one address share its pages, an overlapping one the pages they both cover;
    /// targets there have pages of their own; tiles past the title's memory are each one's own.
    #[test]
    fn textures_share_pages_with_textures_not_with_targets() {
        let (t, va) = (Instant::now(), 0x2_0000_0000usize);
        let mut b = Book::new();
        let mut grow = heaps();
        let a = b.claim(Space::Textures, 1, va, 4, 4, t, &mut grow).unwrap();
        let same = b.claim(Space::Textures, 2, va, 4, 5, t, &mut grow).unwrap();
        assert_eq!(same[..4], a[..], "a texture placed over another one: the same memory");
        assert!(!a.contains(&same[4]), "a tile past the title's memory is its own");
        let over = b.claim(Space::Textures, 3, va + 2 * PAGE, 4, 4, t, &mut grow).unwrap();
        assert_eq!(over[..2], a[2..], "the pages both cover");
        assert!(!a.contains(&over[2]) && !a.contains(&over[3]));
        let target = b.claim(Space::Targets, 4, va, 4, 4, t, &mut grow).unwrap();
        assert!(target.iter().all(|x| !a.contains(x) && !over.contains(x)), "a target's pages are not a texture's");
        // 6 texture pages, 4 target pages, 1 own tile
        assert_eq!(b.bytes(), 11 * PAGE);
    }

    #[test]
    fn unaligned_textures_share_covered_pages_and_keep_neighbours_separate() {
        let now = Instant::now();
        let base = 0x2_0000_0000usize;
        let mut book = Book::new();
        let mut grow = heaps();
        let mut claim = |id, start, bytes, tiles| {
            book.claim(Space::Textures, id, base + start, covered_pages(base + start, bytes), tiles, now, &mut grow).unwrap()
        };
        let a = claim(1, 4096, PAGE, 2);
        let b = claim(2, 8192, PAGE, 2);
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        assert_eq!(a, claim(3, 4096, PAGE, 2));
        let neighbour = claim(4, PAGE + 8192, 4096, 1);
        assert!(!a.contains(&neighbour[0]) && !b.contains(&neighbour[0]));
        assert_eq!(covered_pages(base + 4096, PAGE - 4096), 0);
        assert_eq!(book.bytes(), 4 * PAGE);
    }

    /// The retail game's pattern (#248): textures streamed in over memory used before, nearly
    /// none released. Committed, they held the sum of every texture placed; on pages, the memory
    /// they cover.
    #[test]
    fn streamed_textures_cost_the_memory_they_cover() {
        const SPAN: usize = 256 * MIB;
        let (t, va) = (Instant::now(), 0x3_0000_0000usize);
        let mut b = Book::new();
        let mut grow = heaps();
        let mut placed = 0usize;
        // 8000 textures of 64 KiB to 4 MiB at page addresses all over the span
        let mut x = 0x2545_F491u32;
        for res in 0..8000usize {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let pages = 1usize << (x % 7);
            let at = (x as usize >> 3) % (SPAN / PAGE - pages) * PAGE;
            b.claim(Space::Textures, res, va + at, pages, pages, t, &mut grow).unwrap();
            placed += pages * PAGE;
        }
        assert!(placed > 30 * SPAN, "the test places {placed} bytes");
        assert!(b.bytes() <= SPAN, "{} bytes of pages for a span of {SPAN}", b.bytes());
    }

    /// A page no resource uses keeps its tile (and what it holds) for 2 s, then the tile cools
    /// another 2 s before it is reused; a claim without video memory for all of it keeps nothing.
    #[test]
    fn released_pages_are_kept_then_cooled_before_reuse() {
        let (t, va) = (Instant::now(), 0x4_0000_0000usize);
        let s = Duration::from_secs;
        let mut b = Book::new();
        let mut grow = heaps();
        let first = b.claim(Space::Textures, 1, va, 1, 1, t, &mut grow).unwrap();
        b.release(1, t);
        assert_eq!(b.claim(Space::Textures, 2, va, 1, 1, t + s(1), &mut grow).unwrap(), first, "placed again within 2 s: the same memory");
        b.release(2, t + s(1));
        b.sweep(t + s(2));
        assert!(b.pages.contains_key(&(Space::Textures, va)), "an older retirement cannot expire a reactivated page");
        // the page is let go at 3 s (2 s idle) and its tile cools until 5 s
        b.sweep(t + s(3));
        assert!(b.pages.is_empty());
        b.free.clear();
        assert!(b.claim(Space::Textures, 3, va, 1, 1, t + s(4), &mut || None).is_none(), "the tile is still cooling, no new heap");
        assert_eq!(b.claim(Space::Textures, 4, va + PAGE, 1, 1, t + s(5), &mut || None).unwrap(), first, "cooled down: reused");
        // 3 pages wanted, one tile to be had: nothing kept, the tile reusable
        b.free.push((7, 7));
        assert!(b.claim(Space::Textures, 5, va + 8 * PAGE, 3, 3, t + s(5), &mut || None).is_none());
        assert!(!b.users.contains_key(&5));
        assert!(b.pages.get(&(Space::Textures, va + 8 * PAGE)).is_some_and(|p| p.users == 0), "its page is idle, not in use");
    }

    /// On the GPU: kept-in-step textures placed at one title address are one memory (an upload
    /// through one is what the other reads), and many placements there take one texture's video
    /// memory. Needs a hardware adapter with tiled resources tier 2 (skips otherwise).
    #[test]
    fn title_textures_at_one_address_share_video_memory() {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                eprintln!("SKIPPED title_textures_at_one_address_share_video_memory: software adapter");
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage as usize
            };
            // RGBA8 512x512, one mip: 1 MiB (16 pages) on the console
            let (w, h) = (512u32, 512u32);
            let console_bytes = (w * h * 4) as u64;
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w as u64, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let srv = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_title_texture(dev.as_raw(), va, &desc, srv.0 as u32, console_bytes, riid, &mut p);
                if hr.is_none() {
                    return None;
                }
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "texture at {va:#x}");
                Some(ID3D12Resource::from_raw(p))
            };
            let va = 0x60_0000_0000usize;
            let Some(a) = place_at(va) else {
                eprintln!("SKIPPED title_textures_at_one_address_share_video_memory: no reserved RGBA8 textures");
                return;
            };
            let b = place_at(va).unwrap();

            // texels uploaded through b are what a holds
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            before_execute(&queue);
            let texels: Vec<u8> = (0..w * h * 4).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
            let sub = crate::gpu::texture::Subresource { rows: h as usize, row_bytes: w as usize * 4, depth: 1, data: texels.clone() };
            let failed = crate::gpu::texture::upload_batch(&dev, &queue, &[(b.clone(), vec![sub], vec![srv])]).unwrap();
            assert!(failed.is_empty(), "{failed:?}");
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            dev.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(&a), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            // (tiles shared between two reserved resources: an aliasing barrier between them)
            let alias = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING, Anonymous: D3D12_RESOURCE_BARRIER_0 { Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER { pResourceBefore: std::mem::transmute_copy(&b), pResourceAfter: std::mem::transmute_copy(&a) }) }, ..Default::default() };
            list.ResourceBarrier(&[alias, barrier(srv, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&a), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, srv)]);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            crate::gpu::texture::wait(&dev, &queue).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let pitch = fp.Footprint.RowPitch as usize;
            let bad = (0..h as usize).filter(|&y| std::slice::from_raw_parts((p as *const u8).add(y * pitch), w as usize * 4) != &texels[y * w as usize * 4..(y + 1) * w as usize * 4]).count();
            readback.Unmap(0, None);
            assert_eq!(bad, 0, "rows of the texture placed first that differ from what was uploaded through the second");

            // 256 more placements at 8 addresses: video memory for 8 textures, not 256
            let used0 = local();
            let pages0 = page_bytes().0;
            let more: Vec<ID3D12Resource> = (0..256).map(|k| place_at(va + 0x100_0000 + (k % 8) * 0x10_0000).unwrap()).collect();
            let (used, pages) = (local().saturating_sub(used0), page_bytes().0 - pages0);
            println!("256 RGBA8 512x512 textures at 8 addresses: {pages} bytes of pages, {used} bytes more local video memory");
            assert!(pages <= 8 * console_bytes as usize + 256 * PAGE, "{pages} bytes of pages");
            assert!(used < 64 * console_bytes as usize, "{used} bytes of video memory");
            drop((more, a, b));
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        a.take(va, bytes, align, now, |_, _| Some(n)).unwrap().2
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<u64>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| take(&mut a, a0 + start + k * S, S, PAGE, now)).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for id in live.pop_front().unwrap() {
                    a.release(id, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32, &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }
}
