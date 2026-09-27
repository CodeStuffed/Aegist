//! Seeing the screen, from scratch: what changed between two screenshots,
//! where the buttons and the drawing area are, a fingerprint to recognize a
//! button again, and what shape a stroke left behind.

use serde::{Deserialize, Serialize};

pub type Rgb = [u8; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub fn new(x: i32, y: i32, w: i32, h: i32) -> Rect {
        Rect { x, y, w: w.max(0), h: h.max(0) }
    }
    pub fn right(&self) -> i32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> i32 {
        self.y + self.h
    }
    pub fn area(&self) -> i64 {
        self.w as i64 * self.h as i64
    }
    pub fn center(&self) -> (i32, i32) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }
    pub fn inflate(&self, n: i32) -> Rect {
        Rect::new(self.x - n, self.y - n, self.w + 2 * n, self.h + 2 * n)
    }
    pub fn intersect(&self, o: &Rect) -> Rect {
        let (x, y) = (self.x.max(o.x), self.y.max(o.y));
        Rect::new(x, y, self.right().min(o.right()) - x, self.bottom().min(o.bottom()) - y)
    }
    pub fn union(&self, o: &Rect) -> Rect {
        let (x, y) = (self.x.min(o.x), self.y.min(o.y));
        Rect::new(x, y, self.right().max(o.right()) - x, self.bottom().max(o.bottom()) - y)
    }
    pub fn offset(&self, dx: i32, dy: i32) -> Rect {
        Rect::new(self.x + dx, self.y + dy, self.w, self.h)
    }
}

/// A screenshot: RGB, row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

impl Image {
    pub fn new(w: usize, h: usize, fill: Rgb) -> Image {
        Image { w, h, px: fill.iter().copied().cycle().take(w * h * 3).collect() }
    }

    pub fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.w as i32, self.h as i32)
    }

    pub fn get(&self, x: i32, y: i32) -> Rgb {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return [0, 0, 0];
        }
        let i = (y as usize * self.w + x as usize) * 3;
        [self.px[i], self.px[i + 1], self.px[i + 2]]
    }

    pub fn set(&mut self, x: i32, y: i32, c: Rgb) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return;
        }
        let i = (y as usize * self.w + x as usize) * 3;
        self.px[i..i + 3].copy_from_slice(&c);
    }

    pub fn fill_rect(&mut self, r: Rect, c: Rgb) {
        for y in r.y..r.bottom() {
            for x in r.x..r.right() {
                self.set(x, y, c);
            }
        }
    }

    /// Save as PNG (for showing what the agent saw).
    pub fn save_png(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut enc = png::Encoder::new(f, self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(&self.px)?;
        Ok(())
    }

    /// The most common color in `r` (sampled).
    pub fn modal_color(&self, r: Rect) -> Rgb {
        let r = r.intersect(&self.bounds());
        let step = ((r.area() / 4000) as f64).sqrt().max(1.0) as i32;
        let mut counts: std::collections::HashMap<Rgb, usize> = Default::default();
        for y in (r.y..r.bottom()).step_by(step as usize) {
            for x in (r.x..r.right()).step_by(step as usize) {
                *counts.entry(self.get(x, y)).or_default() += 1;
            }
        }
        counts.into_iter().max_by_key(|(c, n)| (*n, *c)).map_or([0, 0, 0], |(c, _)| c)
    }
}

pub fn color_dist(a: Rgb, b: Rgb) -> i32 {
    (0..3).map(|i| (a[i] as i32 - b[i] as i32).abs()).sum()
}

pub const COLORS: &[(&str, Rgb)] = &[
    ("black", [0, 0, 0]),
    ("white", [255, 255, 255]),
    ("gray", [128, 128, 128]),
    ("red", [230, 30, 30]),
    ("dark red", [136, 0, 21]),
    ("orange", [255, 127, 39]),
    ("yellow", [255, 230, 0]),
    ("green", [34, 177, 76]),
    ("blue", [30, 70, 230]),
    ("light blue", [0, 162, 232]),
    ("purple", [163, 73, 164]),
    ("pink", [255, 150, 200]),
    ("brown", [140, 80, 30]),
];

pub fn color_name(c: Rgb) -> &'static str {
    COLORS.iter().min_by_key(|(_, k)| color_dist(*k, c)).map_or("?", |(n, _)| n)
}

pub fn parse_color(word: &str) -> Option<Rgb> {
    let w = word.trim().to_lowercase();
    let w = match w.as_str() {
        "grey" => "gray",
        "violet" => "purple",
        "cyan" | "sky blue" => "light blue",
        "maroon" => "dark red",
        other => other,
    };
    COLORS.iter().find(|(n, _)| *n == w).map(|(_, c)| *c)
}

/// Close enough to call the same color (by name, or nearly the same value).
pub fn same_color(a: Rgb, b: Rgb) -> bool {
    color_dist(a, b) < 60 || color_name(a) == color_name(b)
}

/// A button's look, to recognize it again: an 8x8 light/dark pattern plus
/// its average color (plain color swatches all have the same pattern).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Print {
    pub shape: u64,
    pub color: Rgb,
    /// The middle half's average color, and whether it's one plain color.
    #[serde(default)]
    pub core: Rgb,
    #[serde(default)]
    pub solid: bool,
}

impl Print {
    pub fn of(img: &Image, r: Rect) -> Print {
        // inside the border, which changes when a button is selected
        let r = if r.w >= 12 && r.h >= 12 { r.inflate(-(r.w.min(r.h) / 8).max(2)) } else { r };
        let r = r.intersect(&img.bounds());
        let mut cells = [0f32; 64];
        let mut sum = [0u64; 3];
        let mut n = 0u64;
        for gy in 0..8 {
            for gx in 0..8 {
                let (x0, x1) = (r.x + r.w * gx / 8, r.x + (r.w * (gx + 1) / 8).max(r.w * gx / 8 + 1));
                let (y0, y1) = (r.y + r.h * gy / 8, r.y + (r.h * (gy + 1) / 8).max(r.h * gy / 8 + 1));
                let (mut g, mut k) = (0f32, 0f32);
                for y in y0..y1.min(r.bottom()) {
                    for x in x0..x1.min(r.right()) {
                        let c = img.get(x, y);
                        g += 0.3 * c[0] as f32 + 0.59 * c[1] as f32 + 0.11 * c[2] as f32;
                        k += 1.0;
                        for i in 0..3 {
                            sum[i] += c[i] as u64;
                        }
                        n += 1;
                    }
                }
                cells[(gy * 8 + gx) as usize] = if k > 0.0 { g / k } else { 0.0 };
            }
        }
        let mean = cells.iter().sum::<f32>() / 64.0;
        let spread = cells.iter().map(|c| (c - mean).abs()).fold(0f32, f32::max);
        let shape = if spread < 12.0 { 0 } else { cells.iter().enumerate().fold(0u64, |a, (i, &c)| a | (((c > mean) as u64) << i)) };
        let n = n.max(1);
        let mid = Rect::new(r.x + r.w / 4, r.y + r.h / 4, (r.w / 2).max(1), (r.h / 2).max(1));
        let pix: Vec<Rgb> = (mid.y..mid.bottom()).flat_map(|y| (mid.x..mid.right()).map(move |x| (x, y))).map(|(x, y)| img.get(x, y)).collect();
        let m = pix.len().max(1) as u64;
        let core = [0, 1, 2].map(|i| (pix.iter().map(|p| p[i] as u64).sum::<u64>() / m) as u8);
        let solid = pix.iter().all(|&p| color_dist(p, core) < 30);
        Print { shape, color: [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8], core, solid }
    }

    pub fn similar(&self, o: &Print) -> bool {
        (self.shape ^ o.shape).count_ones() <= 8 && color_dist(self.color, o.color) < 45
    }

    /// Its middle is one plain color: likely a color swatch.
    pub fn plain(&self) -> bool {
        self.solid
    }
}

/// Pixels that differ between two screenshots, inside `region`.
#[derive(Clone, Debug)]
pub struct Change {
    pub region: Rect,
    /// Region-sized mask, row by row.
    pub mask: Vec<bool>,
    pub count: usize,
    pub bbox: Option<Rect>,
    /// The most common color the changed pixels took.
    pub new_color: Option<Rgb>,
}

impl Change {
    pub fn at(&self, x: i32, y: i32) -> bool {
        if !self.region.contains(x, y) {
            return false;
        }
        self.mask[((y - self.region.y) * self.region.w + (x - self.region.x)) as usize]
    }

    /// Anything changed within `r` pixels of (x, y)?
    pub fn near(&self, x: i32, y: i32, r: i32) -> bool {
        (-r..=r).any(|dy| (-r..=r).any(|dx| self.at(x + dx, y + dy)))
    }

    pub fn fraction_of(&self, r: Rect) -> f32 {
        if r.area() == 0 {
            return 0.0;
        }
        self.count as f32 / r.area() as f32
    }
}

pub fn diff(a: &Image, b: &Image, region: Rect) -> Change {
    let region = region.intersect(&a.bounds()).intersect(&b.bounds());
    let mut mask = vec![false; region.area() as usize];
    let (mut count, mut bbox): (usize, Option<Rect>) = (0, None);
    // the new color that changed the most pixels the most: soft
    // (anti-aliased) edges are part-way to the old color and count for less
    let mut colors: std::collections::HashMap<Rgb, u64> = Default::default();
    for y in region.y..region.bottom() {
        for x in region.x..region.right() {
            let (p, q) = (a.get(x, y), b.get(x, y));
            let d = color_dist(p, q);
            if d > 30 {
                mask[((y - region.y) * region.w + (x - region.x)) as usize] = true;
                count += 1;
                let px = Rect::new(x, y, 1, 1);
                bbox = Some(bbox.map_or(px, |r| r.union(&px)));
                *colors.entry([q[0] & 0xf8, q[1] & 0xf8, q[2] & 0xf8]).or_default() += (d as u64).pow(2);
            }
        }
    }
    let new_color = colors.into_iter().max_by_key(|(c, n)| (*n, *c)).map(|(c, _)| c.map(|v| v | (v >> 5)));
    Change { region, mask, count, bbox, new_color }
}

/// Things that look like buttons in `region` (leaving out `skip`, the
/// canvas): compact clusters of pixels that stand out from the background,
/// 6 to 96 pixels across.
pub fn find_elements(img: &Image, region: Rect, skip: Option<Rect>) -> Vec<Rect> {
    let region = region.intersect(&img.bounds());
    if region.area() == 0 {
        return Vec::new();
    }
    let bg = {
        let mut counts: std::collections::HashMap<Rgb, usize> = Default::default();
        for y in (region.y..region.bottom()).step_by(2) {
            for x in (region.x..region.right()).step_by(2) {
                if !skip.is_some_and(|s| s.contains(x, y)) {
                    *counts.entry(img.get(x, y)).or_default() += 1;
                }
            }
        }
        counts.into_iter().max_by_key(|(c, n)| (*n, *c)).map_or([0, 0, 0], |(c, _)| c)
    };
    let (w, h) = (region.w as usize, region.h as usize);
    let ink: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (region.x + (i % w) as i32, region.y + (i / w) as i32);
            !skip.is_some_and(|s| s.inflate(2).contains(x, y)) && color_dist(img.get(x, y), bg) > 40
        })
        .collect();
    let mut seen = vec![false; w * h];
    let mut found = Vec::new();
    for start in 0..w * h {
        if !ink[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        let mut too_big = false;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            if x1 - x0 > 120 || y1 - y0 > 120 {
                too_big = true;
            }
            // ink within 2 pixels belongs to the same thing
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if ink[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as i32, (y1 - y0 + 1) as i32);
        if !too_big && (6..=96).contains(&bw) && (6..=96).contains(&bh) && bw <= bh * 5 && bh <= bw * 5 {
            found.push(Rect::new(region.x + x0 as i32, region.y + y0 as i32, bw, bh));
        }
    }
    found.sort_by_key(|r| (r.y / 8, r.x));
    found
}

/// The drawing area: the largest region of one plain color inside `window`
/// (at least a sixth of it).
pub fn find_canvas(img: &Image, window: Rect) -> Option<Rect> {
    let window = window.intersect(&img.bounds());
    const STEP: i32 = 4;
    let (gw, gh) = ((window.w / STEP) as usize, (window.h / STEP) as usize);
    if gw == 0 || gh == 0 {
        return None;
    }
    let at = |gx: usize, gy: usize| img.get(window.x + gx as i32 * STEP, window.y + gy as i32 * STEP);
    let mut seen = vec![false; gw * gh];
    let mut best: Option<(usize, Rect)> = None;
    for start in 0..gw * gh {
        if seen[start] {
            continue;
        }
        let color = at(start % gw, start / gw);
        let mut stack = vec![start];
        seen[start] = true;
        let (mut n, mut x0, mut y0, mut x1, mut y1) = (0, usize::MAX, usize::MAX, 0, 0);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % gw, i / gw);
            n += 1;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx as usize >= gw || ny as usize >= gh {
                    continue;
                }
                let j = ny as usize * gw + nx as usize;
                if !seen[j] && color_dist(at(nx as usize, ny as usize), color) <= 12 {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if best.as_ref().is_none_or(|b| n > b.0) {
            let r = Rect::new(window.x + x0 as i32 * STEP, window.y + y0 as i32 * STEP, (x1 - x0 + 1) as i32 * STEP,
                              (y1 - y0 + 1) as i32 * STEP);
            best = Some((n, r));
        }
    }
    let (n, r) = best?;
    // a solid area, not a long thin strip or a scattered background
    let solid = n as f32 * (STEP * STEP) as f32 / r.area().max(1) as f32;
    (n as i64 * (STEP * STEP) as i64 >= window.area() / 6 && solid > 0.6).then(|| r.inflate(-3))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    /// Follows the pointer's path.
    Freehand,
    /// Straight from where the drag started to where it ended.
    Line,
    Rectangle,
    Ellipse,
}

impl Shape {
    pub const ALL: [Shape; 4] = [Shape::Freehand, Shape::Line, Shape::Rectangle, Shape::Ellipse];

    pub fn name(&self) -> &'static str {
        match self {
            Shape::Freehand => "freehand strokes",
            Shape::Line => "straight lines",
            Shape::Rectangle => "rectangles",
            Shape::Ellipse => "ellipses",
        }
    }
}

/// How well a change matches each thing a drag could have drawn: the
/// fraction of points along each candidate outline with changed pixels
/// right next to them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fit {
    pub path: f32,
    pub chord: f32,
    pub boxed: f32,
    pub ellipse: f32,
    /// Changed pixels over the drag's bounding box area.
    pub solid: f32,
}

fn along(a: (i32, i32), b: (i32, i32)) -> impl Iterator<Item = (i32, i32)> {
    let n = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) / 3).max(1);
    (0..=n).map(move |i| (a.0 + (b.0 - a.0) * i / n, a.1 + (b.1 - a.1) * i / n))
}

fn coverage(c: &Change, pts: impl Iterator<Item = (i32, i32)>) -> f32 {
    let (mut hit, mut n) = (0, 0);
    for (x, y) in pts {
        n += 1;
        hit += c.near(x, y, 3) as usize;
    }
    if n == 0 { 0.0 } else { hit as f32 / n as f32 }
}

pub fn fit(c: &Change, path: &[(i32, i32)]) -> Fit {
    let (Some(&s), Some(&e)) = (path.first(), path.last()) else { return Fit::default() };
    let segs: Vec<(i32, i32)> = path.windows(2).flat_map(|w| along(w[0], w[1])).collect();
    let (x0, y0, x1, y1) = (s.0.min(e.0), s.1.min(e.1), s.0.max(e.0), s.1.max(e.1));
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)];
    let edges: Vec<(i32, i32)> = corners.windows(2).flat_map(|w| along(w[0], w[1])).collect();
    let (cx, cy, rx, ry) = ((x0 + x1) as f32 / 2.0, (y0 + y1) as f32 / 2.0, (x1 - x0) as f32 / 2.0, (y1 - y0) as f32 / 2.0);
    let ring = (0..48).map(|i| {
        let t = i as f32 / 48.0 * std::f32::consts::TAU;
        ((cx + rx * t.cos()).round() as i32, (cy + ry * t.sin()).round() as i32)
    });
    let area = ((x1 - x0 + 1) * (y1 - y0 + 1)).max(1) as f32;
    let inside = (y0..=y1).flat_map(|y| (x0..=x1).map(move |x| (x, y))).filter(|&(x, y)| c.at(x, y)).count();
    Fit { path: coverage(c, segs.into_iter()), chord: coverage(c, along(s, e)), boxed: coverage(c, edges.into_iter()),
          ellipse: coverage(c, ring), solid: inside as f32 / area }
}

impl Fit {
    /// Did the change draw `shape` for this drag?
    pub fn matches(&self, shape: Shape) -> bool {
        match shape {
            Shape::Rectangle => self.boxed > 0.8,
            Shape::Ellipse => self.ellipse > 0.75 && self.boxed < 0.6,
            Shape::Line => self.chord > 0.85,
            Shape::Freehand => self.path > 0.85,
        }
    }

    /// The shape an L-shaped probe drag (across the top, then down the
    /// right) drew: a freehand tool follows the L, a line tool cuts the
    /// diagonal, a rectangle tool adds the other two sides, an ellipse
    /// tool rounds it off.
    pub fn probe_shape(&self) -> Option<Shape> {
        if self.boxed > 0.8 {
            Some(Shape::Rectangle)
        } else if self.ellipse > 0.75 && self.boxed < 0.6 {
            Some(Shape::Ellipse)
        } else if self.chord > 0.85 && self.path < 0.7 {
            Some(Shape::Line)
        } else if self.path > 0.85 && self.boxed < 0.75 {
            Some(Shape::Freehand)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(img: &mut Image, a: (i32, i32), b: (i32, i32), c: Rgb) {
        for (x, y) in along(a, b).flat_map(|p| along(p, (p.0 + 1, p.1))) {
            img.set(x, y, c);
        }
        let n = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
        for i in 0..=n {
            img.set(a.0 + (b.0 - a.0) * i / n, a.1 + (b.1 - a.1) * i / n, c);
        }
    }

    #[test]
    fn it_tells_shapes_apart() {
        let blank = Image::new(200, 200, [255; 3]);
        let (s, m, e) = ((40, 50), (160, 50), (160, 150));
        let probe = [s, m, e];
        let red = [230, 30, 30];
        let mut free = blank.clone();
        line(&mut free, s, m, red);
        line(&mut free, m, e, red);
        let mut straight = blank.clone();
        line(&mut straight, s, e, red);
        let mut rect = free.clone();
        line(&mut rect, s, (40, 150), red);
        line(&mut rect, (40, 150), e, red);
        let mut ell = blank.clone();
        for i in 0..400 {
            let t = i as f32 / 400.0 * std::f32::consts::TAU;
            ell.set((100.0 + 60.0 * t.cos()) as i32, (100.0 + 50.0 * t.sin()) as i32, red);
        }
        for (img, want) in [(&free, Shape::Freehand), (&straight, Shape::Line), (&rect, Shape::Rectangle), (&ell, Shape::Ellipse)] {
            let c = diff(&blank, img, blank.bounds());
            assert_eq!(fit(&c, &probe).probe_shape(), Some(want), "{want:?}: {:?}", fit(&c, &probe));
            assert!(fit(&c, &probe).matches(want));
            assert_eq!(color_name(c.new_color.unwrap()), "red");
        }
        assert!(fit(&diff(&blank, &blank, blank.bounds()), &probe).probe_shape().is_none());
    }

    #[test]
    fn it_finds_buttons_and_the_canvas() {
        let mut img = Image::new(400, 300, [200; 3]);
        img.fill_rect(Rect::new(20, 60, 360, 220), [255; 3]);
        for i in 0..5 {
            img.fill_rect(Rect::new(20 + i * 40, 10, 30, 30), [90; 3]);
            img.fill_rect(Rect::new(24 + i * 40, 14, 22, 22), [200; 3]);
            img.fill_rect(Rect::new(30 + i * 40, 20, 3, 10), [20; 3]);
        }
        img.fill_rect(Rect::new(230, 12, 26, 26), [230, 30, 30]);
        let found = find_elements(&img, Rect::new(0, 0, 400, 55), None);
        assert_eq!(find_elements(&img, img.bounds(), Some(Rect::new(20, 60, 360, 220))).len(), 6);
        assert_eq!(found.len(), 6, "{found:?}");
        assert_eq!(found[0], Rect::new(20, 10, 30, 30));
        let canvas = find_canvas(&img, img.bounds()).unwrap();
        assert!(canvas.x >= 20 && canvas.y >= 60 && canvas.right() <= 380 && canvas.w > 340, "{canvas:?}");
        let swatch = Print::of(&img, found[5]);
        assert!(swatch.plain() && color_name(swatch.color) == "red");
        assert!(!Print::of(&img, found[0]).plain() && Print::of(&img, found[0]).similar(&Print::of(&img, found[1])));
        assert_eq!(parse_color("Grey"), Some([128, 128, 128]));
    }
}

