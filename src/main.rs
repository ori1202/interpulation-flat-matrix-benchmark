use wide::f64x4;

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

#[test]
fn compare_factored_vs_wide() {
    use std::time::Instant;
    let q = 500_000_000;

    let m = 50;
    let n = 50;
    let p = 50;

    let a_vals: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let b_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let c_vals: Vec<f64> = (0..p).map(|i| i as f64).collect();

    let mut data = vec![0.0; m * n * p];
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                data[i * n * p + j * p + k] = (i + j + k) as f64;
            }
        }
    }

    // --- Scalar factored ---
    let start = Instant::now();
    let mut sum_scalar = 0.0;
    for qid in 0..q {
        let a = ((qid % (m - 1)) as f64) + 0.3;
        let b = ((qid % (n - 1)) as f64) + 0.4;
        let c = ((qid % (p - 1)) as f64) + 0.5;

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
            ax[lane] = ((idx % (m - 1)) as f64) + 0.3;
            bx[lane] = ((idx % (n - 1)) as f64) + 0.4;
            cx[lane] = ((idx % (p - 1)) as f64) + 0.5;
        }

        let a = f64x4::from(ax);
        let b = f64x4::from(bx);
        let c = f64x4::from(cx);

        // floor to nearest grid index
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

    println!("Scalar factored: {:?}, sum={},    q:{}", t_scalar, sum_scalar,q);
    println!("Wide factored  : {:?}, sum={:?},  q:{}", t_wide, sum_wide.reduce_add(),q);
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




    
    println!("Custom trilinear: {:?} for {} queries", t1, q);
    println!("Factored trilinear: {:?} for {} queries", t_factored, q);
    println!("Nearest neighbor: {:?} for {} queries", t2, q);
    println!("ninterp (linear): {:?} for {} queries", t3, q);
    

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

    const M: usize = 5;
    const N: usize = 5;
    const P: usize = 5;

    let a_vals: [f64; M] = [0.0, 1.0, 2.0, 3.0, 4.0];
    let b_vals: [f64; N] = [0.0, 1.0, 2.0, 3.0, 4.0];
    let c_vals: [f64; P] = [0.0, 1.0, 2.0, 3.0, 4.0];

    // Fill stack 3D array
    let mut data_stack = [[[0.0; P]; N]; M];
    for i in 0..M {
        for j in 0..N {
            for k in 0..P {
                data_stack[i][j][k] = (i + j + k) as f64;
            }
        }
    }

    // Fill heap Vec
    let mut data_heap = vec![0.0; M * N * P];
    for i in 0..M {
        for j in 0..N {
            for k in 0..P {
                data_heap[i * N * P + j * P + k] = (i + j + k) as f64;
            }
        }
    }

    let q = 100_000;

    // Benchmark stack
    let start = Instant::now();
{
    let mut sum: f64 = 0.0;
for qid in 0..q {
    let a = ((qid % (M - 1)) as f64) + 0.3;
    let b = ((qid % (N - 1)) as f64) + 0.4;
    let c = ((qid % (P - 1)) as f64) + 0.5;
    sum += trilinear_interp_stack(a_vals, b_vals, c_vals, data_stack, a, b, c);
}
let t_stack = start.elapsed();
println!("Stack ({} queries): {:?}, sum={}", q, t_stack, sum);
}

    // Benchmark heap
    let start = Instant::now();
    {
    let mut sum: f64 = 0.0;

    for qid in 0..q {
        let a = ((qid % (M - 1)) as f64) + 0.3;
        let b = ((qid % (N - 1)) as f64) + 0.4;
        let c = ((qid % (P - 1)) as f64) + 0.5;
        sum += trilinear_interp(&a_vals, &b_vals, &c_vals, &data_heap, M, N, P, a, b, c);
    }
    let t_heap = start.elapsed();

    println!("Heap ({} queries): {:?}, sum={}", q, t_heap, sum);

}


    // Assert both are roughly the same order of magnitude
    // assert!(
    //     t_stack.as_micros() < t_heap.as_micros() * 5,
    //     "Stack interpolation unexpectedly slow: {:?} vs {:?}",
    //     t_stack,
    //     t_heap
    // );
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

    println!("Naive trilinear   : {:?}, sum={}", t_naive, sum_naive);
    println!("Factored trilinear: {:?}, sum={}", t_factored, sum_factored);
    println!("ninterp (linear)  : {:?}, sum={}", t_ninterp, sum_ninterp);

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
