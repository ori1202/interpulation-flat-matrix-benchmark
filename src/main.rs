use wide::f64x4;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize)]
struct Row {
    a: f64,
    b: f64,
    c: f64,
    value: f64,
}

fn load_interp_table(path: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, usize), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut rows: Vec<Row> = Vec::new();
    for result in rdr.deserialize() {
        let row: Row = result?;
        rows.push(row);
    }

    // Collect unique axis values
    let mut a_vals: Vec<f64> = rows.iter().map(|r| r.a).collect();
    let mut b_vals: Vec<f64> = rows.iter().map(|r| r.b).collect();
    let mut c_vals: Vec<f64> = rows.iter().map(|r| r.c).collect();

    a_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    c_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    a_vals.dedup();
    b_vals.dedup();
    c_vals.dedup();

    let m = a_vals.len();
    let n = b_vals.len();
    let p = c_vals.len();

    // Flatten into Vec<f64> [m × n × p]
    let mut data = vec![0.0; m * n * p];
    for r in rows {
        let i = a_vals.iter().position(|&v| v == r.a).unwrap();
        let j = b_vals.iter().position(|&v| v == r.b).unwrap();
        let k = c_vals.iter().position(|&v| v == r.c).unwrap();
        data[i * n * p + j * p + k] = r.value;
    }

    Ok((a_vals, b_vals, c_vals, data, m, n, p))
}

fn setup_from_csv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, usize) {
    load_interp_table("interp_tb.csv").expect("failed to load CSV")
}


fn trilinear_interp(
    a_vals: &[f64],
    b_vals: &[f64],
    c_vals: &[f64],
    data: &[f64], // flattened 3D array (M×N×K)
    m: usize,
    n: usize,
    p: usize,
    a: f64,
    b: f64,
    c: f64,
) -> f64 {
    let i = a_vals.binary_search_by(|x| x.partial_cmp(&a).unwrap()).unwrap_or_else(|x| x - 1);
    let j = b_vals.binary_search_by(|x| x.partial_cmp(&b).unwrap()).unwrap_or_else(|x| x - 1);
    let k = c_vals.binary_search_by(|x| x.partial_cmp(&c).unwrap()).unwrap_or_else(|x| x - 1);

    let i1 = (i + 1).min(m - 1);
    let j1 = (j + 1).min(n - 1);
    let k1 = (k + 1).min(p - 1);

    let i0 = i.min(m - 2);
    let j0 = j.min(n - 2);
    let k0 = k.min(p - 2);

    let dx = (a - a_vals[i0]) / (a_vals[i1] - a_vals[i0]);
    let dy = (b - b_vals[j0]) / (b_vals[j1] - b_vals[j0]);
    let dz = (c - c_vals[k0]) / (c_vals[k1] - c_vals[k0]);

    let idx = |ii, jj, kk| ii * n * p + jj * p + kk;

    let c000 = data[idx(i0, j0, k0)];
    let c001 = data[idx(i0, j0, k1)];
    let c010 = data[idx(i0, j1, k0)];
    let c011 = data[idx(i0, j1, k1)];
    let c100 = data[idx(i1, j0, k0)];
    let c101 = data[idx(i1, j0, k1)];
    let c110 = data[idx(i1, j1, k0)];
    let c111 = data[idx(i1, j1, k1)];

    c000 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        + c100 * dx * (1.0 - dy) * (1.0 - dz)
        + c010 * (1.0 - dx) * dy * (1.0 - dz)
        + c001 * (1.0 - dx) * (1.0 - dy) * dz
        + c101 * dx * (1.0 - dy) * dz
        + c011 * (1.0 - dx) * dy * dz
        + c110 * dx * dy * (1.0 - dz)
        + c111 * dx * dy * dz
}
 
fn trilinear_interp_factored(
    c000: f64, c001: f64, c010: f64, c011: f64,
    c100: f64, c101: f64, c110: f64, c111: f64,
    dx: f64, dy: f64, dz: f64,
) -> f64 {
    // interpolate along z
    let c00 = c000 * (1.0 - dz) + c001 * dz;
    let c01 = c010 * (1.0 - dz) + c011 * dz;
    let c10 = c100 * (1.0 - dz) + c101 * dz;
    let c11 = c110 * (1.0 - dz) + c111 * dz;

    // along y
    let c0 = c00 * (1.0 - dy) + c01 * dy;
    let c1 = c10 * (1.0 - dy) + c11 * dy;

    // along x
    c0 * (1.0 - dx) + c1 * dx
}

fn trilinear_interp_factored_wide(
    c000: f64x4, c001: f64x4,
    c010: f64x4, c011: f64x4,
    c100: f64x4, c101: f64x4,
    c110: f64x4, c111: f64x4,
    dx: f64x4, dy: f64x4, dz: f64x4,
) -> f64x4 {
    let one = f64x4::splat(1.0);

    // interpolate along z
    let c00 = c000 * (one - dz) + c001 * dz;
    let c01 = c010 * (one - dz) + c011 * dz;
    let c10 = c100 * (one - dz) + c101 * dz;
    let c11 = c110 * (one - dz) + c111 * dz;

    // along y
    let c0 = c00 * (one - dy) + c01 * dy;
    let c1 = c10 * (one - dy) + c11 * dy;

    // along x
    c0 * (one - dx) + c1 * dx
}

fn trilinear_interp_stack<const M: usize, const N: usize, const P: usize>(
    a_vals: [f64; M],
    b_vals: [f64; N],
    c_vals: [f64; P],
    data: [[[f64; P]; N]; M], // stack allocated 3D array
    a: f64,
    b: f64,
    c: f64,
) -> f64 {
    let i = a_vals.iter().position(|&x| x <= a).unwrap_or(M - 2);
    let j = b_vals.iter().position(|&x| x <= b).unwrap_or(N - 2);
    let k = c_vals.iter().position(|&x| x <= c).unwrap_or(P - 2);

    let i1 = (i + 1).min(M - 1);
    let j1 = (j + 1).min(N - 1);
    let k1 = (k + 1).min(P - 1);

    let dx = (a - a_vals[i]) / (a_vals[i1] - a_vals[i]);
    let dy = (b - b_vals[j]) / (b_vals[j1] - b_vals[j]);
    let dz = (c - c_vals[k]) / (c_vals[k1] - c_vals[k]);

    let c000 = data[i][j][k];
    let c001 = data[i][j][k1];
    let c010 = data[i][j1][k];
    let c011 = data[i][j1][k1];
    let c100 = data[i1][j][k];
    let c101 = data[i1][j][k1];
    let c110 = data[i1][j1][k];
    let c111 = data[i1][j1][k1];

    c000 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        + c100 * dx * (1.0 - dy) * (1.0 - dz)
        + c010 * (1.0 - dx) * dy * (1.0 - dz)
        + c001 * (1.0 - dx) * (1.0 - dy) * dz
        + c101 * dx * (1.0 - dy) * dz
        + c011 * (1.0 - dx) * dy * dz
        + c110 * dx * dy * (1.0 - dz)
        + c111 * dx * dy * dz
}

fn nearest_neighbor_interp(
    a_vals: &[f64],
    b_vals: &[f64],
    c_vals: &[f64],
    d_vals: &[f64],
    a: f64,
    b: f64,
    c: f64,
) -> f64 {
    let mut best_dist = f64::INFINITY;
    let mut best_val = 0.0;

    for i in 0..a_vals.len() {
        let da = a_vals[i] - a;
        let db = b_vals[i] - b;
        let dc = c_vals[i] - c;
        let dist = da * da + db * db + dc * dc;
        if dist < best_dist {
            best_dist = dist;
            best_val = d_vals[i];
        }
    }
    best_val
}

/// Perform 6D linear interpolation over flattened data.
fn hexalinear_interp_clamp(
    a_vals: &[f64],
    b_vals: &[f64],
    c_vals: &[f64],
    d_vals: &[f64],
    e_vals: &[f64],
    f_vals: &[f64],
    data: &[f64],              // flattened 6D array
    m: usize,
    n: usize,
    o: usize,
    p: usize,
    q: usize,
    r: usize,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
) -> f64 {
    // --- find indices and fractions ---
    let idx_clamp = |vals: &[f64], v: f64| {
        let mut i = vals
            .binary_search_by(|x| x.partial_cmp(&v).unwrap())
            .unwrap_or_else(|x| x.saturating_sub(1));
        if i >= vals.len() - 1 {
            i = vals.len() - 2;
        }
        (i, i + 1, (v - vals[i]) / (vals[i + 1] - vals[i]))
    };

    let (i0, i1, dx) = idx_clamp(a_vals, a);
    let (j0, j1, dy) = idx_clamp(b_vals, b);
    let (k0, k1, dz) = idx_clamp(c_vals, c);
    let (l0, l1, dw) = idx_clamp(d_vals, d);
    let (m0, m1, du) = idx_clamp(e_vals, e);
    let (n0, n1, dv) = idx_clamp(f_vals, f);

    // --- flatten index ---
    let idx = |ia, jb, kc, ld, me, nf| (((((ia * n + jb) * o + kc) * p + ld) * q + me) * r + nf);

    let mut acc = 0.0;

    for (ai, aw) in [(i0, 1.0 - dx), (i1, dx)] {
        for (bj, bw) in [(j0, 1.0 - dy), (j1, dy)] {
            for (ck, cw) in [(k0, 1.0 - dz), (k1, dz)] {
                for (dl, dwv) in [(l0, 1.0 - dw), (l1, dw)] {
                    for (em, ew) in [(m0, 1.0 - du), (m1, du)] {
                        for (fn_, fw) in [(n0, 1.0 - dv), (n1, dv)] {
                            let w = aw * bw * cw * dwv * ew * fw;
                            acc += data[idx(ai, bj, ck, dl, em, fn_)] * w;
                        }
                    }
                }
            }
        }
    }
    acc
}


/// Return (i0, i1, t) where i0<=x<i1 on `axis`, and t = (x-x0)/(x1-x0).
fn bracket(axis: &[f64], x: f64) -> (usize, usize, f64) {
    let n = axis.len();
    assert!(n >= 2, "axis must have at least 2 points");
    let i = match axis.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(idx) => idx.min(n - 2),
        Err(ins) => if ins == 0 { 0 } else { (ins - 1).min(n - 2) },
    };
    let i1 = i + 1;
    let (x0, x1) = (axis[i], axis[i1]);
    let t = if x1 != x0 { (x - x0) / (x1 - x0) } else { 0.0 };
    (i, i1, t)
}

/// Hexalinear (6-D) interpolation over a regular grid stored in a flat Vec<f64>.
/// Axes must be sorted ascending. `dims = [d0,d1,d2,d3,d4,d5]` and
/// `data.len() == d0*d1*d2*d3*d4*d5`. `x = [x0..x5]` is the query.
fn hexalinear_interp_fast(
    axes: [&[f64]; 6],
    data: &[f64],
    dims: [usize; 6],
    x: [f64; 6],
) -> f64 {
    // bracketing
    let (i0a, i1a, ta) = bracket(axes[0], x[0]);
    let (i0b, i1b, tb) = bracket(axes[1], x[1]);
    let (i0c, i1c, tc) = bracket(axes[2], x[2]);
    let (i0d, i1d, td) = bracket(axes[3], x[3]);
    let (i0e, i1e, te) = bracket(axes[4], x[4]);
    let (i0f, i1f, tf) = bracket(axes[5], x[5]);

    let [_d0, d1, d2, d3, d4, d5] = dims;
    let s5 = 1;
    let s4 = d5 * s5;
    let s3 = d4 * s4;
    let s2 = d3 * s3;
    let s1 = d2 * s2;
    let s0 = d1 * s1;

    // base index helper
    let base = |ia, ib, ic, id, ie| ia * s0 + ib * s1 + ic * s2 + id * s3 + ie * s4;

    // 1) interpolate along F
    let mut f32 = [0.0; 32];
    let mut ptr = 0;
    for &ia in &[i0a, i1a] {
        for &ib in &[i0b, i1b] {
            for &ic in &[i0c, i1c] {
                for &id in &[i0d, i1d] {
                    for &ie in &[i0e, i1e] {
                        let b = base(ia, ib, ic, id, ie);
                        let v0 = data[b + i0f * s5];
                        let v1 = data[b + i1f * s5];
                        f32[ptr] = v0 + (v1 - v0) * tf;
                        ptr += 1;
                    }
                }
            }
        }
    }

    // 2) reduce along E
    let mut f16 = [0.0; 16];
    for i in 0..16 { f16[i] = f32[2*i] + (f32[2*i+1] - f32[2*i]) * te; }

    // 3) reduce along D
    let mut f8 = [0.0; 8];
    for i in 0..8 { f8[i] = f16[2*i] + (f16[2*i+1] - f16[2*i]) * td; }

    // 4) reduce along C
    let mut f4 = [0.0; 4];
    for i in 0..4 { f4[i] = f8[2*i] + (f8[2*i+1] - f8[2*i]) * tc; }

    // 5) reduce along B
    let mut f2 = [0.0; 2];
    for i in 0..2 { f2[i] = f4[2*i] + (f4[2*i+1] - f4[2*i]) * tb; }

    // 6) reduce along A
    f2[0] + (f2[1] - f2[0]) * ta
}

#[test]
fn hexalinear_known_linear_function_is_exact() {
        // axes: 0,1,2 on each dimension
        let ax = vec![0.0, 1.0, 2.0];
        let bx = vec![0.0, 1.0, 2.0];
        let cx = vec![0.0, 1.0, 2.0];
        let dx = vec![0.0, 1.0, 2.0];
        let ex = vec![0.0, 1.0, 2.0];
        let fx = vec![0.0, 1.0, 2.0];
        let axes = [&ax[..], &bx[..], &cx[..], &dx[..], &ex[..], &fx[..]];
        let dims = [3, 3, 3, 3, 3, 3];

        // fill data[i,j,k,l,m,n] = a_i + b_j + c_k + d_l + e_m + f_n
        let [d0, d1, d2, d3, d4, d5] = dims;
        let s5 = 1usize;
        let s4 = d5 * s5;
        let s3 = d4 * s4;
        let s2 = d3 * s3;
        let s1 = d2 * s2;
        let s0 = d1 * s1;
        let mut data = vec![0.0; d0 * d1 * d2 * d3 * d4 * d5];
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    for l in 0..d3 {
                        for m in 0..d4 {
                            for n in 0..d5 {
                                let idx = i * s0 + j * s1 + k * s2 + l * s3 + m * s4 + n * s5;
                                data[idx] = ax[i] + bx[j] + cx[k] + dx[l] + ex[m] + fx[n];
                            }
                        }
                    }
                }
            }
        }
       

        // a few exact tests (grid points + midpoints)
        let cases = [
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
            ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 6.0),
            ([2.0, 2.0, 2.0, 2.0, 1.0, 2.0], 11.0),
            ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 3.0),
            ([1.2, 0.3, 1.5, 0.1, 1.8, 0.4], 1.2 + 0.3 + 1.5 + 0.1 + 1.8 + 0.4),
        ];

        for (pt, expect) in cases {
            let v = hexalinear_interp_fast(axes, &data, dims, pt);
            assert!(
                (v - expect).abs() < 1e-12,
                "hexalinear mismatch at {:?}: got {}, expected {}",
                pt,
                v,
                expect
            );
        }
    }


/// Hexalinear (6-D) interpolation over a regular grid stored in a flat Vec<f64>.
/// Axes must be sorted ascending. `dims = [d0,d1,d2,d3,d4,d5]` and
/// `data.len() == d0*d1*d2*d3*d4*d5`. `x = [x0..x5]` is the query.
fn _hexalinear_interp(axes: [&[f64]; 6], data: &[f64], dims: [usize; 6], x: [f64; 6]) -> f64 {
    // bracketing + weights
    let (i0a, i1a, ta) = bracket(axes[0], x[0]);
    let (i0b, i1b, tb) = bracket(axes[1], x[1]);
    let (i0c, i1c, tc) = bracket(axes[2], x[2]);
    let (i0d, i1d, td) = bracket(axes[3], x[3]);
    let (i0e, i1e, te) = bracket(axes[4], x[4]);
    let (i0f, i1f, tf) = bracket(axes[5], x[5]);

    // strides for row-major [a,b,c,d,e,f]
    let [d0, d1, d2, d3, d4, d5] = dims;
    assert_eq!(data.len(), d0 * d1 * d2 * d3 * d4 * d5, "data size mismatch");
    let s5 = 1usize;
    let s4 = d5 * s5;
    let s3 = d4 * s4;
    let s2 = d3 * s3;
    let s1 = d2 * s2;
    let s0 = d1 * s1;

    // 1) reduce along F for all 32 combos of A..E (bit order: E,D,C,B,A as 0..4)
    let mut v32 = [0.0f64; 32];
    for code in 0..32 {
        let ia = if (code >> 4) & 1 == 0 { i0a } else { i1a };
        let ib = if (code >> 3) & 1 == 0 { i0b } else { i1b };
        let ic = if (code >> 2) & 1 == 0 { i0c } else { i1c };
        let id = if (code >> 1) & 1 == 0 { i0d } else { i1d };
        let ie = if (code >> 0) & 1 == 0 { i0e } else { i1e };
        let base = ia * s0 + ib * s1 + ic * s2 + id * s3 + ie * s4;
        let v0 = data[base + i0f * s5];
        let v1 = data[base + i1f * s5];
        v32[code] = v0 + (v1 - v0) * tf;
    }

    // 2) successive linear reductions: along E, D, C, B, then A
    let mut v16 = [0.0f64; 16];
    for i in 0..16 { v16[i] = v32[2 * i] + (v32[2 * i + 1] - v32[2 * i]) * te; }

    let mut v8 = [0.0f64; 8];
    for i in 0..8 { v8[i] = v16[2 * i] + (v16[2 * i + 1] - v16[2 * i]) * td; }

    let mut v4 = [0.0f64; 4];
    for i in 0..4 { v4[i] = v8[2 * i] + (v8[2 * i + 1] - v8[2 * i]) * tc; }

    let mut v2 = [0.0f64; 2];
    for i in 0..2 { v2[i] = v4[2 * i] + (v4[2 * i + 1] - v4[2 * i]) * tb; }

    v2[0] + (v2[1] - v2[0]) * ta
}


fn idx6(n:usize,o:usize,p:usize,q:usize,r:usize,
    ia:usize,jb:usize,kc:usize,ld:usize,me:usize,nf:usize) -> usize {
((((ia*n + jb)*o + kc)*p + ld)*q + me)*r + nf
}


#[test]
fn compare_hexalinear_speeds() {
    use std::time::Instant;

    // Number of points per axis (change this to scale problem size)
    let npts = 12; // try 10, 15, 30 … careful with memory usage!
    let a_vals: Vec<f64> = (0..npts).map(|i| i as f64).collect();
    let b_vals = a_vals.clone();
    let c_vals = a_vals.clone();
    let d_vals = a_vals.clone();
    let e_vals = a_vals.clone();
    let f_vals = a_vals.clone();

    let (m, n, o, p, q, r) = (npts, npts, npts, npts, npts, npts);

    // Fill data with f(a,b,c,d,e,f) = sum of indices
    let mut data = vec![0.0; m * n * o * p * q * r];
    let idx = |ia, jb, kc, ld, me, nf| (((((ia * n + jb) * o + kc) * p + ld) * q + me) * r + nf);
    for ia in 0..m {
        for jb in 0..n {
            for kc in 0..o {
                for ld in 0..p {
                    for me in 0..q {
                        for nf in 0..r {
                            data[idx(ia, jb, kc, ld, me, nf)] =
                                ia as f64 + jb as f64 + kc as f64 +
                                ld as f64 + me as f64 + nf as f64;
                        }
                    }
                }
            }
        }
    }

    let axes = [&a_vals[..], &b_vals[..], &c_vals[..], &d_vals[..], &e_vals[..], &f_vals[..]];
    let dims = [m, n, o, p, q, r];

    // Deterministic queries
    let nq_total = 10_000*50; // reduce if needed
    let queries: Vec<[f64; 6]> = (0..nq_total)
        .map(|k| {
            let t = (k as f64) / (nq_total as f64) * ((npts - 1) as f64);
            [
                t,
                (npts - 1) as f64 - t,
                t * 0.5,
                (npts as f64) / 2.0,
                (k % npts) as f64,
                (k % (npts / 2).max(1)) as f64 + 0.25,
            ]
        })
        .collect();

     // --- Memory usage (MB) ---
     let data_bytes = data.len() * std::mem::size_of::<f64>();
     let queries_bytes = queries.len() * std::mem::size_of::<[f64; 6]>();
     println!(
        "Memory usage: data = {:.2} MB ({} values), queries = {:.2} MB ({} queries × 6 values)",
        data_bytes as f64 / (1024.0 * 1024.0),
        data.len(),
        queries_bytes as f64 / (1024.0 * 1024.0),
        queries.len(),
    );

    let nq = queries.len() as f64;

    // --- Benchmark hexalinear_interp_clamp ---
    let start = Instant::now();
    let mut sum1 = 0.0;
    for qv in &queries {
        sum1 += hexalinear_interp_clamp(
            &a_vals, &b_vals, &c_vals, &d_vals, &e_vals, &f_vals,
            &data, m, n, o, p, q, r, qv[0], qv[1], qv[2], qv[3], qv[4], qv[5],
        );
    }
    let t1 = start.elapsed();

    // --- Benchmark hexalinear_interp_fast ---
    let start = Instant::now();
    let mut sum2 = 0.0;
    for qv in &queries {
        sum2 += hexalinear_interp_fast(axes, &data, dims, *qv);
    }
    let t2 = start.elapsed();

    // --- Results ---
    println!(
        "hexalinear_interp_clamp: {:?} (≈{:.3} ns/query), sum={}",
        t1,
        (t1.as_nanos() as f64) / nq,
        sum1
    );
    println!(
        "hexalinear_interp_fast : {:?} (≈{:.3} ns/query), sum={}",
        t2,
        (t2.as_nanos() as f64) / nq,
        sum2
    );

    // Sanity check
    assert!(
        (sum1 - sum2).abs() < 1e-9,
        "Mismatch: {} vs {}",
        sum1,
        sum2
    );
}




#[test]
fn test_hexalinear_known_function() {
    // Axes
    let a_vals = vec![0.0, 1.0, 2.0];
    let b_vals = vec![0.0, 1.0, 2.0];
    let c_vals = vec![0.0, 1.0, 2.0];
    let d_vals = vec![0.0, 1.0, 2.0];
    let e_vals = vec![0.0, 1.0, 2.0];
    let f_vals = vec![0.0, 1.0, 2.0];

    let (m, n, o, p, q, r) = (
        a_vals.len(),
        b_vals.len(),
        c_vals.len(),
        d_vals.len(),
        e_vals.len(),
        f_vals.len(),
    );

    // f(a,b,c,d,e,f) = a+b+c+d+e+f
    let mut data = vec![0.0; m * n * o * p * q * r];
    let idx = |ia, jb, kc, ld, me, nf| (((((ia * n + jb) * o + kc) * p + ld) * q + me) * r + nf);
    for ia in 0..m {
        for jb in 0..n {
            for kc in 0..o {
                for ld in 0..p {
                    for me in 0..q {
                        for nf in 0..r {
                            data[idx(ia, jb, kc, ld, me, nf)] = a_vals[ia]
                                + b_vals[jb]
                                + c_vals[kc]
                                + d_vals[ld]
                                + e_vals[me]
                                + f_vals[nf];
                        }
                    }
                }
            }
        }
    }
    // --- Print the interpolation table in CSV format ---
    println!("a,b,c,d,e,f,value");
    for ia in 0..m {
        for jb in 0..n {
            for kc in 0..o {
                for ld in 0..p {
                    for me in 0..q {
                        for nf in 0..r {
                            let v = data[idx(ia, jb, kc, ld, me, nf)];
                            println!(
                                "{},{},{},{},{},{},{}",
                                a_vals[ia],
                                b_vals[jb],
                                c_vals[kc],
                                d_vals[ld],
                                e_vals[me],
                                f_vals[nf],
                                v
                            );
                        }
                    }
                }
            }
        }
    }

    // Test points with expected = sum of coordinates
    let test_points = vec![
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 6.0),
        ([2.0, 2.0, 2.0, 0f64, 2.0, 4.0], 12.0),
        ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 3.0),
        ([1.5, 0.5, 0.0, 2.0, 0.0, 1.0], 5.0),
        ([0.2, 1.8, 0.7, 0.3, 1.1, 1.9], 6.0),
        ([1.25, 0.75, 1.5, 0.5, 1.0, 0.0], 5.0),
    ];

    for (pt, expected) in test_points {
        let val = hexalinear_interp_clamp(
            &a_vals,
            &b_vals,
            &c_vals,
            &d_vals,
            &e_vals,
            &f_vals,
            &data,
            m,
            n,
            o,
            p,
            q,
            r,
            pt[0],
            pt[1],
            pt[2],
            pt[3],
            pt[4],
            pt[5],
        );
        println!("point={:?}, expected={}, got={}", pt, expected, val);
        assert!(
            (val - expected).abs() < 1e-9,
            "Mismatch at {:?}: expected {}, got {}",
            pt,
            expected,
            val
        );
    }
}



#[test]
fn test_hexalinear_random_with_known_checks() {
    use rand::{SeedableRng, Rng};
    use rand::rngs::SmallRng;

    // 3 points per axis → 3^6 = 729 grid nodes
    let a_vals = vec![0.0, 1.0, 2.0];
    let b_vals = vec![0.0, 1.0, 2.0];
    let c_vals = vec![0.0, 1.0, 2.0];
    let d_vals = vec![0.0, 1.0, 2.0];
    let e_vals = vec![0.0, 1.0, 2.0];
    let f_vals = vec![0.0, 1.0, 2.0];
    let (m,n,o,p,q,r) = (3,3,3,3,3,3);

    // deterministic random table in [-20, 20)
    let mut rng = SmallRng::seed_from_u64(2);
    let mut data = vec![0.0; m*n*o*p*q*r];
    for ia in 0..m {
        for jb in 0..n {
            for kc in 0..o {
                for ld in 0..p {
                    for me in 0..q {
                        for nf in 0..r {
                            let v = rng.gen_range(-20.0..20.0);
                            data[idx6(n,o,p,q,r, ia,jb,kc,ld,me,nf)] = v;
                        }
                    }
                }
            }
        }
    }

    // ---- Pin specific corners to known values for precise checks ----
    // 1) Line along F at (A,B,C,D,E)=(0,0,0,0,1): F=1 → -5, F=2 → 10
    {
        let ia=0; let jb=0; let kc=0; let ld=0; let me=1;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,me,1)] = -5.0;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,me,2)] = 10.0;
    }
    // 2) Bilinear patch on (E,F) at (A,B,C,D)=(0,0,0,0):
    //    (E,F)=(1,1)->-5, (1,2)->10, (2,1)->7, (2,2)->3
    {
        let ia=0; let jb=0; let kc=0; let ld=0;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,0,0)] = 5.0;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,0,1)] = 10.0;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,0,2)] =  12.0;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,1,0)] =  10.0;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,1,1)] =  5.0;
        data[idx6(n,o,p,q,r, ia,jb,kc,ld,1,2)] =  -5.0;
    }

    // ---- Print a CSV slice for (A,B,C,D)=(0,0,0,0) over E,F ----
    println!("a,b,c,d,e,f,value   (slice @ a=0,b=0,c=0,d=0)");
    let (ia,jb,kc,ld) = (0,0,0,0);
    for me in 0..q {
        for nf in 0..r {
            let v = data[idx6(n,o,p,q,r, ia,jb,kc,ld,me,nf)];
            println!("{},{},{},{},{},{},{}",
                a_vals[ia], b_vals[jb], c_vals[kc], d_vals[ld],
                e_vals[me], f_vals[nf], v
            );
        }
    }

    // ---- Checks ----

    // (A,B,C,D,E) fixed at (0,0,0,0,1); interpolate along F at F=1.5.
    // Expect exact linear mix between -5 (F=1) and 10 (F=2): (-5+10)/2 = 2.5
    {
        let val = hexalinear_interp_clamp(
            &a_vals,&b_vals,&c_vals,&d_vals,&e_vals,&f_vals,&data,
            m,n,o,p,q,r, 0.0,0.0,0.0,0.0,1.0,1.5
        );
        println!("Check F-only at (0,0,0,0,1,1.5) → {}", val);
        // assert!((val - 2.5).abs() < 1e-9,
        //     "F-line lerp failed: expected 2.5, got {}", val);
    }

    // Bilinear on (E,F) plane at (A,B,C,D)=(0,0,0,0): query (E=1.25,F=1.75)
    // Corners: v11=-5, v12=10, v21=7, v22=3 → bilerp with te=0.25, tf=0.75
    {
        let te = 0.25; let tf = 0.75;
        let v11 = -5.0; let v12 = 10.0;
        let v21 =  7.0; let v22 =  3.0;
        let v_e1 = v11 + (v12 - v11)*tf;
        let v_e2 = v21 + (v22 - v21)*tf;
        let expect = v_e1 + (v_e2 - v_e1)*te; // 0.25/0.75 bilerp
        let val = hexalinear_interp_clamp(
            &a_vals,&b_vals,&c_vals,&d_vals,&e_vals,&f_vals,&data,
            m,n,o,p,q,r, 0.0,0.0,0.0,0.0,-0.99,6.
        );
        println!("Check EF-bilinear at (0.0,0.0,0.0,0.0,1.1,1.333339) → {} (expect {})", val, expect);
        assert!((val - expect).abs() < 1e-9,
            "EF bilinear failed: expected {}, got {}", expect, val);
    }

    // General convex-hull bound: pick a point inside cell (1,1,1,1,1,1) with offsets
    // Interpolated value must lie within [min, max] of that cell's 64 corner values.
    {
        let a=1.3; let b=1.7; let c=1.4; let d=1.2; let e=1.6; let f=1.1;
        let v = hexalinear_interp_clamp(
            &a_vals,&b_vals,&c_vals,&d_vals,&e_vals,&f_vals,&data,
            m,n,o,p,q,r, a,b,c,d,e,f
        );

        let corners = [(0,0,0,0,0,0),(0,0,0,0,0,1),(0,0,0,0,1,0),(0,0,0,0,1,1),
                       (0,0,0,1,0,0),(0,0,0,1,0,1),(0,0,0,1,1,0),(0,0,0,1,1,1),
                       (0,0,1,0,0,0),(0,0,1,0,0,1),(0,0,1,0,1,0),(0,0,1,0,1,1),
                       (0,0,1,1,0,0),(0,0,1,1,0,1),(0,0,1,1,1,0),(0,0,1,1,1,1),
                       (0,1,0,0,0,0),(0,1,0,0,0,1),(0,1,0,0,1,0),(0,1,0,0,1,1),
                       (0,1,0,1,0,0),(0,1,0,1,0,1),(0,1,0,1,1,0),(0,1,0,1,1,1),
                       (0,1,1,0,0,0),(0,1,1,0,0,1),(0,1,1,0,1,0),(0,1,1,0,1,1),
                       (0,1,1,1,0,0),(0,1,1,1,0,1),(0,1,1,1,1,0),(0,1,1,1,1,1),
                       (1,0,0,0,0,0),(1,0,0,0,0,1),(1,0,0,0,1,0),(1,0,0,0,1,1),
                       (1,0,0,1,0,0),(1,0,0,1,0,1),(1,0,0,1,1,0),(1,0,0,1,1,1),
                       (1,0,1,0,0,0),(1,0,1,0,0,1),(1,0,1,0,1,0),(1,0,1,0,1,1),
                       (1,0,1,1,0,0),(1,0,1,1,0,1),(1,0,1,1,1,0),(1,0,1,1,1,1),
                       (1,1,0,0,0,0),(1,1,0,0,0,1),(1,1,0,0,1,0),(1,1,0,0,1,1),
                       (1,1,0,1,0,0),(1,1,0,1,0,1),(1,1,0,1,1,0),(1,1,0,1,1,1),
                       (1,1,1,0,0,0),(1,1,1,0,0,1),(1,1,1,0,1,0),(1,1,1,0,1,1),
                       (1,1,1,1,0,0),(1,1,1,1,0,1),(1,1,1,1,1,0),(1,1,1,1,1,1)];
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for (da,db,dc,dd,de,df) in corners {
            let ia=1+da; let jb=1+db; let kc=1+dc; let ld=1+dd; let me=1+de; let nf=1+df;
            let vv = data[idx6(n,o,p,q,r, ia,jb,kc,ld,me,nf)];
            vmin = vmin.min(vv); vmax = vmax.max(vv);
        }
        println!("Convex-hull check: v={}, min={}, max={}", v, vmin, vmax);
        assert!(v >= vmin - 1e-12 && v <= vmax + 1e-12,
            "Interpolated value {} outside [{}, {}]", v, vmin, vmax);
    }
}




#[test]
fn compare_factored_vs_wide() {
    use std::time::Instant;
    let q = 500_000;

    let (a_vals, b_vals, c_vals, data, m, n, p) = setup_from_csv();

    // --- Scalar factored ---
    let start = Instant::now();
    let mut sum_scalar = 0.0;
    for qid in 0..q {
        let a = a_vals[(qid % (m - 1))] + 0.3;
        let b = b_vals[(qid % (n - 1))] + 0.4;
        let c = c_vals[(qid % (p - 1))] + 0.5;

        let i = a.floor() as usize;
        let j = b.floor() as usize;
        let k = c.floor() as usize;

        let dx = a - a_vals[i];
        let dy = b - b_vals[j];
        let dz = c - c_vals[k];

        let idx = |ii, jj, kk| ii * n * p + jj * p + kk;

        sum_scalar += trilinear_interp_factored(
            data[idx(i, j, k)],
            data[idx(i, j, k + 1)],
            data[idx(i, j + 1, k)],
            data[idx(i, j + 1, k + 1)],
            data[idx(i + 1, j, k)],
            data[idx(i + 1, j, k + 1)],
            data[idx(i + 1, j + 1, k)],
            data[idx(i + 1, j + 1, k + 1)],
            dx,
            dy,
            dz,
        );
    }
    let t_scalar = start.elapsed();

    // --- Wide factored ---
    let start = Instant::now();
    let mut sum_wide = f64x4::splat(0.0);
    for qid in (0..q).step_by(4) {
        let mut ax = [0.0; 4];
        let mut bx = [0.0; 4];
        let mut cx = [0.0; 4];
        for lane in 0..4 {
            let idx = qid + lane;
            if idx >= q { break; }
            ax[lane] = a_vals[(idx % (m - 1))] + 0.3;
            bx[lane] = b_vals[(idx % (n - 1))] + 0.4;
            cx[lane] = c_vals[(idx % (p - 1))] + 0.5;
        }
        let a = f64x4::from(ax);
        let b = f64x4::from(bx);
        let c = f64x4::from(cx);

        let i: [usize; 4] = a.to_array().map(|v| v.floor() as usize);
        let j: [usize; 4] = b.to_array().map(|v| v.floor() as usize);
        let k: [usize; 4] = c.to_array().map(|v| v.floor() as usize);

        let dx = a - f64x4::from([a_vals[i[0]], a_vals[i[1]], a_vals[i[2]], a_vals[i[3]]]);
        let dy = b - f64x4::from([b_vals[j[0]], b_vals[j[1]], b_vals[j[2]], b_vals[j[3]]]);
        let dz = c - f64x4::from([c_vals[k[0]], c_vals[k[1]], c_vals[k[2]], c_vals[k[3]]]);

        let idx = |ii, jj, kk| ii * n * p + jj * p + kk;

        let c000 = f64x4::from([data[idx(i[0], j[0], k[0])], data[idx(i[1], j[1], k[1])], data[idx(i[2], j[2], k[2])], data[idx(i[3], j[3], k[3])]]);
        let c001 = f64x4::from([data[idx(i[0], j[0], k[0] + 1)], data[idx(i[1], j[1], k[1] + 1)], data[idx(i[2], j[2], k[2] + 1)], data[idx(i[3], j[3], k[3] + 1)]]);
        let c010 = f64x4::from([data[idx(i[0], j[0] + 1, k[0])], data[idx(i[1], j[1] + 1, k[1])], data[idx(i[2], j[2] + 1, k[2])], data[idx(i[3], j[3] + 1, k[3])]]);
        let c011 = f64x4::from([data[idx(i[0], j[0] + 1, k[0] + 1)], data[idx(i[1], j[1] + 1, k[1] + 1)], data[idx(i[2], j[2] + 1, k[2] + 1)], data[idx(i[3], j[3] + 1, k[3] + 1)]]);
        let c100 = f64x4::from([data[idx(i[0] + 1, j[0], k[0])], data[idx(i[1] + 1, j[1], k[1])], data[idx(i[2] + 1, j[2], k[2])], data[idx(i[3] + 1, j[3], k[3])]]);
        let c101 = f64x4::from([data[idx(i[0] + 1, j[0], k[0] + 1)], data[idx(i[1] + 1, j[1], k[1] + 1)], data[idx(i[2] + 1, j[2], k[2] + 1)], data[idx(i[3] + 1, j[3], k[3] + 1)]]);
        let c110 = f64x4::from([data[idx(i[0] + 1, j[0] + 1, k[0])], data[idx(i[1] + 1, j[1] + 1, k[1])], data[idx(i[2] + 1, j[2] + 1, k[2])], data[idx(i[3] + 1, j[3] + 1, k[3])]]);
        let c111 = f64x4::from([data[idx(i[0] + 1, j[0] + 1, k[0] + 1)], data[idx(i[1] + 1, j[1] + 1, k[1] + 1)], data[idx(i[2] + 1, j[2] + 1, k[2] + 1)], data[idx(i[3] + 1, j[3] + 1, k[3] + 1)]]);

        sum_wide += trilinear_interp_factored_wide(c000, c001, c010, c011, c100, c101, c110, c111, dx, dy, dz);
    }
    let t_wide = start.elapsed();

    println!("Scalar factored: {:?}, per query: {:?}, sum={},    q:{}", t_scalar,t_scalar/q as u32   ,   sum_scalar, q);
    println!("Wide factored  : {:?}, per query: {:?}, sum={:?},  q:{}", t_wide,t_wide    /q as u32   ,     sum_wide.reduce_add(), q);
}


#[cfg(debug_assertions)]
#[test]
fn compare_interpolation_speeds_with_ninterp() {
    use std::time::Instant;
    use ninterp::prelude::*;

    let m = 50;
    let n = 50;
    let p = 50;

    let a_vals: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let b_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let c_vals: Vec<f64> = (0..p).map(|i| i as f64).collect();

    // Fill data cube (flattened in row-major order)
    let mut data = vec![0.0; m * n * p];
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                data[i * n * p + j * p + k] = (i + j + k) as f64;
            }
        }
    }

    // Flatten for nearest neighbor
    let mut a_flat = Vec::new();
    let mut b_flat = Vec::new();
    let mut c_flat = Vec::new();
    let mut d_flat = Vec::new();
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                a_flat.push(a_vals[i]);
                b_flat.push(b_vals[j]);
                c_flat.push(c_vals[k]);
                d_flat.push(data[i * n * p + j * p + k]);
            }
        }
    }

    // Prepare ninterp interpolator
    let interp3d = Interp3D::new(
        ndarray::Array1::from(a_vals.clone()),
        ndarray::Array1::from(b_vals.clone()),
        ndarray::Array1::from(c_vals.clone()),
        ndarray::Array::from_shape_vec((m, n, p), data.clone()).unwrap(),
        strategy::Linear,
        Extrapolate::Error, // fail if outside domain
    )
    .expect("Interp3D::new should succeed");

    let q = 5000;

    // --- Benchmark ninterp ---
    let start = Instant::now();
    {for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;
        let _ = interp3d.interpolate(&[a, b, c]).unwrap();
    }}
    let t3 = start.elapsed();

    // --- Benchmark custom trilinear ---
    let start = Instant::now();
    {for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;
        let _ = trilinear_interp(&a_vals, &b_vals, &c_vals, &data, m, n, p, a, b, c);
    }}
    let t1 = start.elapsed();


    // --- Benchmark heap factorized ---
    let start = Instant::now();
    for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;

        // find cell indices
        let i = a.floor() as usize;
        let j = b.floor() as usize;
        let k = c.floor() as usize;

        let dx = a - a_vals[i];
        let dy = b - b_vals[j];
        let dz = c - c_vals[k];

        let idx = |ii, jj, kk| ii * n * p + jj * p + kk;

        _ = trilinear_interp_factored(
            data[idx(i, j, k)],
            data[idx(i, j, k + 1)],
            data[idx(i, j + 1, k)],
            data[idx(i, j + 1, k + 1)],
            data[idx(i + 1, j, k)],
            data[idx(i + 1, j, k + 1)],
            data[idx(i + 1, j + 1, k)],
            data[idx(i + 1, j + 1, k + 1)],
            dx,
            dy,
            dz,
        );
        }
        let t_factored = start.elapsed();



    // --- Benchmark nearest neighbor ---
    let start = Instant::now();
    {for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;
        let _ = nearest_neighbor_interp(&a_flat, &b_flat, &c_flat, &d_flat, a, b, c);
    }}
    let t2 = start.elapsed();




    
    println!("Custom trilinear: {:?} for {} queries  , per query: {:?}", t1, q,t1/q as u32);
    println!("Factored trilinear: {:?} for {} queries, per query: {:?}", t_factored, q,t_factored/q as u32);
    println!("Nearest neighbor: {:?} for {} queries  , per query: {:?}", t2, q,t2/q as u32);
    println!("ninterp (linear): {:?} for {} queries  , per query: {:?}", t3, q,t3/q as u32);
    

    // --- Assertions with messages ---
    assert!(
        t1.as_micros() < 150,
        "Custom trilinear too slow: {:?}, expected <150µs",
        t1
    );
    assert!(
        t2.as_millis() > 300 && t2.as_millis() < 700,
        "Nearest neighbor out of expected range: {:?}, expected ~400–600ms",
        t2
    );
    assert!(
        t3 > t1,
        "ninterp should not be faster than custom trilinear: custom={:?}, ninterp={:?}",
        t1,
        t3
    );
}

#[test]
fn test_trilinear_known_answers() {
    use ninterp::prelude::*;

    let a_vals = vec![0.0, 1.0, 2.0];
    let b_vals = vec![0.0, 1.0, 2.0];
    let c_vals = vec![0.0, 1.0, 2.0];

    // Define f(a, b, c) = a + b + c
    let mut data = vec![0.0; a_vals.len() * b_vals.len() * c_vals.len()];
    for i in 0..a_vals.len() {
        for j in 0..b_vals.len() {
            for k in 0..c_vals.len() {
                data[i * b_vals.len() * c_vals.len() + j * c_vals.len() + k] =
                    a_vals[i] + b_vals[j] + c_vals[k];
            }
        }
    }

    // Build ninterp interpolator
    let interp3d = Interp3D::new(
        ndarray::Array1::from(a_vals.clone()),
        ndarray::Array1::from(b_vals.clone()),
        ndarray::Array1::from(c_vals.clone()),
        ndarray::Array::from_shape_vec((a_vals.len(), b_vals.len(), c_vals.len()), data.clone()).unwrap(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .expect("Interp3D::new should succeed");

    // Test points with known answers
    let test_points = vec![
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 1.0], 3.0),
        ([2.0, 2.0, 2.0], 6.0),
        ([0.5, 0.5, 0.5], 1.5),
        ([1.5, 0.5, 0.5], 2.5),
    ];

    for (pt, expected) in test_points {
        let val_custom = trilinear_interp(&a_vals, &b_vals, &c_vals, &data, 3, 3, 3, pt[0], pt[1], pt[2]);
        let val_ninterp = interp3d.interpolate(&pt).unwrap();

        println!("point={:?}, expected={}, custom={}, ninterp={}", pt, expected, val_custom, val_ninterp);

        assert!(
            (val_custom - expected).abs() < 1e-9,
            "Custom trilinear mismatch at {:?}: got {}, expected {}",
            pt,
            val_custom,
            expected
        );
        assert!(
            (val_ninterp - expected).abs() < 1e-9,
            "ninterp mismatch at {:?}: got {}, expected {}",
            pt,
            val_ninterp,
            expected
        );
    }
}

#[test]
fn compare_stack_vs_heap() {
    use std::time::Instant;


    let (a_vals, b_vals, c_vals, data, m, n, p) = setup_from_csv();

    let q = 100_000;

    // Benchmark heap version
    let start = Instant::now();
    let mut sum_heap = 0.0;
    for qid in 0..q {
        let a = a_vals[qid % (m - 1)]+0.3;
        let b = b_vals[qid % (n - 1)]+0.4;
        let c = c_vals[qid % (p - 1)]+0.5;
        sum_heap += trilinear_interp(&a_vals, &b_vals, &c_vals, &data, m, n, p, a, b, c);
    }
    let t_heap = start.elapsed();
 
    println!(
        "Heap ({} queries): {:?}, per query: {:?}, sum={}",
        q,
        t_heap,
        t_heap / q as u32, // per-query
        sum_heap
    );

    // ---------------- Stack benchmark ----------------
    // Use the same size as heap but as const generics
    const M: usize = 50;
    const N: usize = 50;
    const P: usize = 50;

    let mut a_vals: [f64; M] = [0.0; M];
    let mut b_vals: [f64; N] = [0.0; N];
    let mut c_vals: [f64; P] = [0.0; P];

    for i in 0..M { a_vals[i] = i as f64; }
    for j in 0..N { b_vals[j] = j as f64; }
    for k in 0..P { c_vals[k] = k as f64; }

    let mut data_stack = [[[0.0; P]; N]; M];
    for i in 0..M {
        for j in 0..N {
            for k in 0..P {
                data_stack[i][j][k] = (i + j + k) as f64;
            }
        }
    }

    let start = Instant::now();
    let mut sum_stack = 0.0;
    for qid in 0..q {
        let a = ((qid % (M - 1)) as f64) + 0.3;
        let b = ((qid % (N - 1)) as f64) + 0.4;
        let c = ((qid % (P - 1)) as f64) + 0.5;
        sum_stack += trilinear_interp_stack(a_vals, b_vals, c_vals, data_stack, a, b, c);
    }
    let t_stack = start.elapsed();
    let per_stack_ns = t_stack.as_nanos() as f64 / q as f64;
    println!(
        "Stack ({} queries): {:?}, per query: {:.2} ns, sum={}",
        q, t_stack, per_stack_ns, sum_stack
    );
// println!("Stack ({} queries): {:?}, per query: {:?} sum={}", q, t_stack,t_stack/q as u32, sum);



    // Assert both are roughly the same order of magnitude
    assert!(
        t_stack.as_micros() < t_heap.as_micros() * 5,
        "Stack interpolation unexpectedly slow: {:?} vs {:?}",
        t_stack,
        t_heap
    );
}

#[test]
fn compare_trilinear_variants() {
    use std::time::Instant;
    use ninterp::prelude::*;

    let m = 50;
    let n = 50;
    let p = 50;

    let a_vals: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let b_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let c_vals: Vec<f64> = (0..p).map(|i| i as f64).collect();

    // Fill data cube (flattened)
    let mut data = vec![0.0; m * n * p];
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                data[i * n * p + j * p + k] = (i + j + k) as f64;
            }
        }
    }

    // Setup ninterp
    let interp3d = Interp3D::new(
        ndarray::Array1::from(a_vals.clone()),
        ndarray::Array1::from(b_vals.clone()),
        ndarray::Array1::from(c_vals.clone()),
        ndarray::Array::from_shape_vec((m, n, p), data.clone()).unwrap(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .expect("Interp3D::new should succeed");

    let q = 5000000;

    // --- Benchmark heap naive ---
    let start = Instant::now();
    let mut sum_naive = 0.0;
    for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;
        sum_naive += trilinear_interp(&a_vals, &b_vals, &c_vals, &data, m, n, p, a, b, c);
    }
    let t_naive = start.elapsed();

    // --- Benchmark heap factorized ---
    let start = Instant::now();
    let mut sum_factored = 0.0;
    for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;

        // find cell indices
        let i = a.floor() as usize;
        let j = b.floor() as usize;
        let k = c.floor() as usize;

        let dx = a - a_vals[i];
        let dy = b - b_vals[j];
        let dz = c - c_vals[k];

        let idx = |ii, jj, kk| ii * n * p + jj * p + kk;

        sum_factored += trilinear_interp_factored(
            data[idx(i, j, k)],
            data[idx(i, j, k + 1)],
            data[idx(i, j + 1, k)],
            data[idx(i, j + 1, k + 1)],
            data[idx(i + 1, j, k)],
            data[idx(i + 1, j, k + 1)],
            data[idx(i + 1, j + 1, k)],
            data[idx(i + 1, j + 1, k + 1)],
            dx,
            dy,
            dz,
        );
    }
    let t_factored = start.elapsed();

    // --- Benchmark ninterp ---
    let start = Instant::now();
    let mut sum_ninterp = 0.0;
    for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;
        sum_ninterp += interp3d.interpolate(&[a, b, c]).unwrap();
    }
    let t_ninterp = start.elapsed();

    println!("Naive trilinear   : {:?},per query: {:?}, sum={}", t_naive,t_naive            /q as u32, sum_naive);
    println!("Factored trilinear: {:?},per query: {:?}, sum={}", t_factored, t_factored     /q as u32, sum_factored);
    println!("ninterp (linear)  : {:?},per query: {:?}, sum={}", t_ninterp,t_ninterp        /q as u32, sum_ninterp);

    // --- Assertions ---
    assert!(
        (sum_naive - sum_factored).abs() < 1e-9,
        "Naive vs Factored mismatch: {} vs {}",
        sum_naive,
        sum_factored
    );
    assert!(
        (sum_naive - sum_ninterp).abs() < 1e-9,
        "Naive vs ninterp mismatch: {} vs {}",
        sum_naive,
        sum_ninterp
    );
}

// fn generate_interp_csv() {
//     use std::fs::File;
//     use std::io::Write;

//     const M: usize = 50;
//     const N: usize = 50;
//     const P: usize = 50;

//     let mut file = File::create("interp_tb.csv").expect("failed to create csv");
//     writeln!(file, "a,b,c,value").unwrap();

//     for i in 0..M {
//         for j in 0..N {
//             for k in 0..P {
//                 let value = (i + j + k) as f64;
//                 writeln!(file, "{},{},{},{}", i, j, k, value).unwrap();
//             }
//         }
//     }
// }