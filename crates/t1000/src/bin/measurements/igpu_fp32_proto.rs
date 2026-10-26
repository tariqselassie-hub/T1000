// SPDX-License-Identifier: MIT OR Apache-2.0
//! FP32 iGPU prototype: device-resident OpenCL statevector vs rayon FP32 CPU reference.
//!
//! Circuit per layer: Ry on every qubit, then one fused diagonal pass (Rz on every qubit
//! plus ring CZ). Usage: `igpu_fp32_proto [min_q] [max_q] [depth] [max_cpu_q]`

use rayon::prelude::*;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::time::Instant;
use t1000::gpu::opencl::*;

const CL_MEM_READ_ONLY: ClUlong = 1 << 2;
const CL_MEM_ALLOC_HOST_PTR: ClUlong = 1 << 4;
const CL_MEM_COPY_HOST_PTR: ClUlong = 1 << 5;

const SRC: &str = r#"
__kernel void init_kernel(__global float* re, __global float* im) {
    size_t i = get_global_id(0);
    re[i] = (i == 0) ? 1.0f : 0.0f;
    im[i] = 0.0f;
}
__kernel void ry_kernel(__global float* re, __global float* im,
                        const ulong stride, const float c, const float s) {
    size_t id = get_global_id(0);
    size_t k = id % stride;
    size_t i0 = (id / stride) * (stride * 2) + k;
    size_t i1 = i0 + stride;
    float r0 = re[i0], m0 = im[i0], r1 = re[i1], m1 = im[i1];
    re[i0] = c * r0 - s * r1;  im[i0] = c * m0 - s * m1;
    re[i1] = s * r0 + c * r1;  im[i1] = s * m0 + c * m1;
}
__kernel void diag_kernel(__global float* re, __global float* im,
                          __constant float* half_ang, const uint n) {
    size_t idx = get_global_id(0);
    float phase = 0.0f;
    for (uint q = 0; q < n; q++) {
        float h = half_ang[q];
        phase += ((idx >> q) & 1) ? h : -h;
    }
    uint rot = (uint)((idx >> 1) | ((idx & 1) << (n - 1)));
    if (popcount((uint)idx & rot) & 1) phase += 3.14159265358979f;
    float cs;
    float sn = sincos(phase, &cs);
    float r = re[idx], m = im[idx];
    re[idx] = r * cs - m * sn;
    im[idx] = r * sn + m * cs;
}
"#;

fn angles(depth: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * std::f32::consts::TAU
    };
    let ry = (0..depth * n).map(|_| next()).collect();
    let rz = (0..depth * n).map(|_| next()).collect();
    (ry, rz)
}

fn cpu_run(n: usize, depth: usize, ry: &[f32], rz: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let dim = 1usize << n;
    let mut re = vec![0f32; dim];
    let mut im = vec![0f32; dim];
    re[0] = 1.0;
    for l in 0..depth {
        for q in 0..n {
            let stride = 1usize << q;
            let (s, c) = (ry[l * n + q] * 0.5).sin_cos();
            re.par_chunks_mut(2 * stride)
                .zip(im.par_chunks_mut(2 * stride))
                .for_each(|(r, m)| {
                    for k in 0..stride {
                        let (r0, m0, r1, m1) = (r[k], m[k], r[k + stride], m[k + stride]);
                        r[k] = c * r0 - s * r1;
                        m[k] = c * m0 - s * m1;
                        r[k + stride] = s * r0 + c * r1;
                        m[k + stride] = s * m0 + c * m1;
                    }
                });
        }
        let half: Vec<f32> = (0..n).map(|q| rz[l * n + q] * 0.5).collect();
        re.par_iter_mut()
            .zip(im.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (r, m))| {
                let mut phase = 0f32;
                for (q, h) in half.iter().enumerate() {
                    phase += if (idx >> q) & 1 == 1 { *h } else { -*h };
                }
                let rot = (idx >> 1) | ((idx & 1) << (n - 1));
                if (idx & rot).count_ones() & 1 == 1 {
                    phase += std::f32::consts::PI;
                }
                let (sn, cs) = phase.sin_cos();
                let (a, b) = (*r, *m);
                *r = a * cs - b * sn;
                *m = a * sn + b * cs;
            });
    }
    (re, im)
}

struct Gpu {
    d: OpenClDispatch,
    ctx: ClContext,
    queue: ClCommandQueue,
    init: ClKernel,
    ry: ClKernel,
    diag: ClKernel,
    name: String,
}

unsafe fn setup() -> Option<Gpu> {
    let d = OpenClDispatch::load()?;
    let mut np = 0u32;
    (d.clGetPlatformIDs)(0, ptr::null_mut(), &mut np);
    let mut ps = vec![ptr::null_mut(); np as usize];
    (d.clGetPlatformIDs)(np, ps.as_mut_ptr(), ptr::null_mut());
    for &p in &ps {
        let mut nd = 0u32;
        if (d.clGetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, 0, ptr::null_mut(), &mut nd) != CL_SUCCESS
            || nd == 0
        {
            continue;
        }
        let mut devs = vec![ptr::null_mut(); nd as usize];
        (d.clGetDeviceIDs)(
            p,
            CL_DEVICE_TYPE_GPU,
            nd,
            devs.as_mut_ptr(),
            ptr::null_mut(),
        );
        let dev = devs[0];
        let mut nb = [0u8; 256];
        (d.clGetDeviceInfo)(
            dev,
            CL_DEVICE_NAME,
            256,
            nb.as_mut_ptr() as *mut c_void,
            ptr::null_mut(),
        );
        let name = CStr::from_ptr(nb.as_ptr() as *const c_char)
            .to_string_lossy()
            .to_string();
        let mut err = 0i32;
        let ctx = (d.clCreateContext)(ptr::null(), 1, &dev, None, ptr::null_mut(), &mut err);
        if err != CL_SUCCESS {
            return None;
        }
        let queue = (d.clCreateCommandQueue)(ctx, dev, 0, &mut err);
        let src = CString::new(SRC).ok()?;
        let sp = src.as_ptr();
        let prog = (d.clCreateProgramWithSource)(ctx, 1, &sp, ptr::null(), &mut err);
        let b = (d.clBuildProgram)(prog, 1, &dev, ptr::null(), None, ptr::null_mut());
        if b != CL_SUCCESS {
            eprintln!("OpenCL build failed: {b}");
            return None;
        }
        let mk = |n: &str| {
            let c = CString::new(n).unwrap();
            let mut e = 0i32;
            (d.clCreateKernel)(prog, c.as_ptr(), &mut e)
        };
        let (init, ry, diag) = (mk("init_kernel"), mk("ry_kernel"), mk("diag_kernel"));
        return Some(Gpu {
            d,
            ctx,
            queue,
            init,
            ry,
            diag,
            name,
        });
    }
    None
}

unsafe fn arg<T>(g: &Gpu, k: ClKernel, i: u32, v: &T) {
    (g.d.clSetKernelArg)(
        k,
        i,
        std::mem::size_of::<T>(),
        v as *const T as *const c_void,
    );
}

unsafe fn run(
    g: &Gpu,
    n: usize,
    depth: usize,
    ry: &[f32],
    rz: &[f32],
    readback: bool,
) -> (f64, Option<(Vec<f32>, Vec<f32>)>) {
    let dim = 1usize << n;
    let mut e = 0i32;
    let re = (g.d.clCreateBuffer)(
        g.ctx,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        dim * 4,
        ptr::null_mut(),
        &mut e,
    );
    assert_eq!(e, 0, "alloc re");
    let im = (g.d.clCreateBuffer)(
        g.ctx,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        dim * 4,
        ptr::null_mut(),
        &mut e,
    );
    assert_eq!(e, 0, "alloc im");
    let half: Vec<f32> = rz.iter().map(|a| a * 0.5).collect();
    let hb = (g.d.clCreateBuffer)(
        g.ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        half.len() * 4,
        half.as_ptr() as *mut c_void,
        &mut e,
    );
    assert_eq!(e, 0, "alloc ang");
    let gws = dim;
    let enq = |k: ClKernel, w: usize| {
        let r = (g.d.clEnqueueNDRangeKernel)(
            g.queue,
            k,
            1,
            ptr::null(),
            &w,
            ptr::null(),
            0,
            ptr::null(),
            ptr::null_mut(),
        );
        assert_eq!(r, 0, "enqueue {r}");
    };
    (g.d.clFinish)(g.queue);
    let t = Instant::now();
    arg(g, g.init, 0, &re);
    arg(g, g.init, 1, &im);
    enq(g.init, gws);
    for l in 0..depth {
        for q in 0..n {
            let stride = 1u64 << q;
            let (s, c) = (ry[l * n + q] * 0.5).sin_cos();
            arg(g, g.ry, 0, &re);
            arg(g, g.ry, 1, &im);
            arg(g, g.ry, 2, &stride);
            arg(g, g.ry, 3, &c);
            arg(g, g.ry, 4, &s);
            enq(g.ry, dim / 2);
        }
        // Per-layer slice of the angle table is passed by creating a sub-view via offset buffer:
        // simplest correct approach is a dedicated small buffer per layer.
        let layer_buf = (g.d.clCreateBuffer)(
            g.ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            n * 4,
            half[l * n..].as_ptr() as *mut c_void,
            &mut e,
        );
        arg(g, g.diag, 0, &re);
        arg(g, g.diag, 1, &im);
        arg(g, g.diag, 2, &layer_buf);
        arg(g, g.diag, 3, &(n as u32));
        enq(g.diag, gws);
        (g.d.clFinish)(g.queue);
        (g.d.clReleaseMemObject)(layer_buf);
    }
    (g.d.clFinish)(g.queue);
    let secs = t.elapsed().as_secs_f64();
    let out = if readback {
        let mut hr = vec![0f32; dim];
        let mut hi = vec![0f32; dim];
        (g.d.clEnqueueReadBuffer)(
            g.queue,
            re,
            CL_TRUE,
            0,
            dim * 4,
            hr.as_mut_ptr() as *mut c_void,
            0,
            ptr::null(),
            ptr::null_mut(),
        );
        (g.d.clEnqueueReadBuffer)(
            g.queue,
            im,
            CL_TRUE,
            0,
            dim * 4,
            hi.as_mut_ptr() as *mut c_void,
            0,
            ptr::null(),
            ptr::null_mut(),
        );
        Some((hr, hi))
    } else {
        None
    };
    (g.d.clReleaseMemObject)(hb);
    (g.d.clReleaseMemObject)(re);
    (g.d.clReleaseMemObject)(im);
    (secs, out)
}

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let (lo, hi) = (*a.first().unwrap_or(&12), *a.get(1).unwrap_or(&26));
    let depth = *a.get(2).unwrap_or(&12);
    let max_cpu = *a.get(3).unwrap_or(&26);
    let g = unsafe { setup() }.expect("FP32 OpenCL setup failed");
    println!(
        "Device: {} | depth {depth} | FP32 SoA, device-resident",
        g.name
    );

    // Correctness: GPU vs CPU FP32 at a small size.
    let n = 12;
    let (ry, rz) = angles(depth, n);
    let (cr, ci) = cpu_run(n, depth, &ry, &rz);
    let (_, out) = unsafe { run(&g, n, depth, &ry, &rz, true) };
    let (gr, gi) = out.unwrap();
    let maxdiff = cr
        .iter()
        .zip(&gr)
        .chain(ci.iter().zip(&gi))
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let norm: f32 = gr.iter().zip(&gi).map(|(x, y)| x * x + y * y).sum();
    println!("Check N={n}: max|GPU-CPU| = {maxdiff:.2e}, GPU norm = {norm:.6}\n");

    println!(
        "{:>3} {:>9} {:>10} {:>10} {:>8}",
        "N", "state_MB", "cpu32_s", "gpu32_s", "speedup"
    );
    for n in (lo..=hi).step_by(2) {
        let (ry, rz) = angles(depth, n);
        let mb = (1usize << n) * 8 / (1 << 20);
        let gpu = unsafe { run(&g, n, depth, &ry, &rz, false) }.0;
        if n <= max_cpu {
            let t = Instant::now();
            let _ = cpu_run(n, depth, &ry, &rz);
            let cpu = t.elapsed().as_secs_f64();
            println!("{n:>3} {mb:>9} {cpu:>10.3} {gpu:>10.3} {:>7.2}x", cpu / gpu);
        } else {
            println!("{n:>3} {mb:>9} {:>10} {gpu:>10.3}", "-");
        }
    }
}
