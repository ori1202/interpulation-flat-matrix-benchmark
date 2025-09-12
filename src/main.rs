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
