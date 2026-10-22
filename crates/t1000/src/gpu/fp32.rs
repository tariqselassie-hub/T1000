// SPDX-License-Identifier: MIT OR Apache-2.0
//! FP32 OpenCL circuit engine for GPUs without FP64 (e.g. Intel UHD / Xe-LP).
//!
//! The statevector is converted to FP32, uploaded once, evolved entirely on the device
//! through every op of a compiled circuit, then downloaded and widened back to FP64.
//! Results therefore carry FP32 rounding (about 1e-7 relative per amplitude).

use super::opencl::*;
use crate::compiler::{CircuitOp, CompiledCircuit};
use crate::state::QuantumState;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

const CL_MEM_COPY_HOST_PTR: ClUlong = 1 << 5;

/// Indices are 32-bit on the device, so dense registers are capped below 2^30 amplitudes.
const MAX_QUBITS: usize = 29;

const SRC: &str = r#"
inline uint ins0(uint x, uint b) {
    return ((x >> b) << (b + 1)) | (x & ((1u << b) - 1u));
}
__kernel void u1q(__global float* re, __global float* im, const uint q,
                  const float a_r, const float a_i, const float b_r, const float b_i,
                  const float c_r, const float c_i, const float d_r, const float d_i) {
    uint i0 = ins0((uint)get_global_id(0), q);
    uint i1 = i0 | (1u << q);
    float r0 = re[i0], m0 = im[i0], r1 = re[i1], m1 = im[i1];
    re[i0] = a_r * r0 - a_i * m0 + b_r * r1 - b_i * m1;
    im[i0] = a_r * m0 + a_i * r0 + b_r * m1 + b_i * r1;
    re[i1] = c_r * r0 - c_i * m0 + d_r * r1 - d_i * m1;
    im[i1] = c_r * m0 + c_i * r0 + d_r * m1 + d_i * r1;
}
__kernel void perm2(__global float* re, __global float* im, const uint lo, const uint hi,
                    const uint off_a, const uint off_b) {
    uint base = ins0(ins0((uint)get_global_id(0), lo), hi);
    uint a = base | off_a, b = base | off_b;
    float tr = re[a], ti = im[a];
    re[a] = re[b]; im[a] = im[b];
    re[b] = tr;    im[b] = ti;
}
__kernel void cphase(__global float* re, __global float* im, const uint mask,
                     const float c, const float s) {
    uint i = (uint)get_global_id(0);
    if ((i & mask) == mask) {
        float r = re[i], m = im[i];
        re[i] = r * c - m * s;
        im[i] = r * s + m * c;
    }
}
"#;

/// FP32 GPU backend. Not `Send`/`Sync` (raw OpenCL handles).
#[allow(missing_debug_implementations)]
pub struct Fp32GpuEngine {
    d: OpenClDispatch,
    ctx: ClContext,
    queue: ClCommandQueue,
    program: ClProgram,
    k_u1q: ClKernel,
    k_perm2: ClKernel,
    k_cphase: ClKernel,
    pub device_name: String,
}

impl Fp32GpuEngine {
    /// Mounts the first OpenCL GPU and builds the FP32 kernels; `None` if unavailable.
    pub fn new() -> Option<Self> {
        let d = OpenClDispatch::load()?;
        unsafe {
            let mut np = 0u32;
            if (d.clGetPlatformIDs)(0, ptr::null_mut(), &mut np) != CL_SUCCESS || np == 0 {
                return None;
            }
            let mut ps = vec![ptr::null_mut(); np as usize];
            (d.clGetPlatformIDs)(np, ps.as_mut_ptr(), ptr::null_mut());
            for &p in &ps {
                let mut nd = 0u32;
                if (d.clGetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, 0, ptr::null_mut(), &mut nd)
                    != CL_SUCCESS
                    || nd == 0
                {
                    continue;
                }
                let mut devs = vec![ptr::null_mut(); nd as usize];
                (d.clGetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, nd, devs.as_mut_ptr(), ptr::null_mut());
                let dev = devs[0];
                let mut nb = [0u8; 256];
                (d.clGetDeviceInfo)(
                    dev,
                    CL_DEVICE_NAME,
                    256,
                    nb.as_mut_ptr() as *mut c_void,
                    ptr::null_mut(),
                );
                let device_name = CStr::from_ptr(nb.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .to_string();

                let mut err = 0i32;
                let ctx = (d.clCreateContext)(ptr::null(), 1, &dev, None, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS || ctx.is_null() {
                    return None;
                }
                let queue = (d.clCreateCommandQueue)(ctx, dev, 0, &mut err);
                if err != CL_SUCCESS || queue.is_null() {
                    (d.clReleaseContext)(ctx);
                    return None;
                }
                let src = CString::new(SRC).ok()?;
                let sp = src.as_ptr();
                let program = (d.clCreateProgramWithSource)(ctx, 1, &sp, ptr::null(), &mut err);
                if err != CL_SUCCESS
                    || program.is_null()
                    || (d.clBuildProgram)(program, 1, &dev, ptr::null(), None, ptr::null_mut())
                        != CL_SUCCESS
                {
                    (d.clReleaseCommandQueue)(queue);
                    (d.clReleaseContext)(ctx);
                    return None;
                }
                let mk = |name: &str| {
                    let c = CString::new(name).unwrap();
                    let mut e = 0i32;
                    let k = (d.clCreateKernel)(program, c.as_ptr(), &mut e);
                    (e == CL_SUCCESS && !k.is_null()).then_some(k)
                };
                let (k_u1q, k_perm2, k_cphase) = (mk("u1q")?, mk("perm2")?, mk("cphase")?);
                return Some(Self { d, ctx, queue, program, k_u1q, k_perm2, k_cphase, device_name });
            }
        }
        None
    }

    /// Largest register the engine accepts.
    pub const fn max_qubits() -> usize {
        MAX_QUBITS
    }

    /// `true` if every op in the circuit has an FP32 device kernel.
    pub fn supports(circuit: &CompiledCircuit) -> bool {
        circuit.num_qubits <= MAX_QUBITS
            && circuit
                .stages
                .iter()
                .flat_map(|s| s.ops.iter())
                .all(|op| !matches!(op, CircuitOp::RademacherPhase { .. }))
    }

    unsafe fn arg<T>(&self, k: ClKernel, i: u32, v: &T) {
        (self.d.clSetKernelArg)(k, i, std::mem::size_of::<T>(), v as *const T as *const c_void);
    }

    unsafe fn launch(&self, k: ClKernel, work: usize) -> Result<(), String> {
        let r = (self.d.clEnqueueNDRangeKernel)(
            self.queue, k, 1, ptr::null(), &work, ptr::null(), 0, ptr::null(), ptr::null_mut(),
        );
        if r == CL_SUCCESS { Ok(()) } else { Err(format!("clEnqueueNDRangeKernel failed: {r}")) }
    }

    unsafe fn u1q(&self, re: &ClMem, im: &ClMem, dim: usize, q: usize, m: [f32; 8]) -> Result<(), String> {
        let k = self.k_u1q;
        self.arg(k, 0, re);
        self.arg(k, 1, im);
        self.arg(k, 2, &(q as u32));
        for (i, v) in m.iter().enumerate() {
            self.arg(k, 3 + i as u32, v);
        }
        self.launch(k, dim / 2)
    }

    unsafe fn perm2(&self, re: &ClMem, im: &ClMem, dim: usize, qa: usize, qb: usize, off_a: u32, off_b: u32) -> Result<(), String> {
        let (lo, hi) = (qa.min(qb) as u32, qa.max(qb) as u32);
        let k = self.k_perm2;
        self.arg(k, 0, re);
        self.arg(k, 1, im);
        self.arg(k, 2, &lo);
        self.arg(k, 3, &hi);
        self.arg(k, 4, &off_a);
        self.arg(k, 5, &off_b);
        self.launch(k, dim / 4)
    }

    unsafe fn cphase(&self, re: &ClMem, im: &ClMem, dim: usize, mask: u32, theta: f64) -> Result<(), String> {
        let (s, c) = (theta.sin() as f32, theta.cos() as f32);
        let k = self.k_cphase;
        self.arg(k, 0, re);
        self.arg(k, 1, im);
        self.arg(k, 2, &mask);
        self.arg(k, 3, &c);
        self.arg(k, 4, &s);
        self.launch(k, dim)
    }

    unsafe fn apply(&self, re: &ClMem, im: &ClMem, dim: usize, op: &CircuitOp) -> Result<(), String> {
        match op {
            CircuitOp::Ry { qubit, theta } => {
                let (s, c) = ((theta * 0.5).sin() as f32, (theta * 0.5).cos() as f32);
                self.u1q(re, im, dim, *qubit, [c, 0.0, -s, 0.0, s, 0.0, c, 0.0])
            }
            CircuitOp::Rz { qubit, theta } => {
                let (s, c) = ((theta * 0.5).sin() as f32, (theta * 0.5).cos() as f32);
                self.u1q(re, im, dim, *qubit, [c, -s, 0.0, 0.0, 0.0, 0.0, c, s])
            }
            CircuitOp::Unitary1Q { qubit, matrix: u } => self.u1q(
                re, im, dim, *qubit,
                [u.u00_re as f32, u.u00_im as f32, u.u01_re as f32, u.u01_im as f32,
                 u.u10_re as f32, u.u10_im as f32, u.u11_re as f32, u.u11_im as f32],
            ),
            CircuitOp::CZ { q1, q2 } => self.cphase(re, im, dim, (1u32 << q1) | (1u32 << q2), std::f64::consts::PI),
            CircuitOp::ControlledPhase { control, target, theta } => {
                self.cphase(re, im, dim, (1u32 << control) | (1u32 << target), *theta)
            }
            CircuitOp::CNOT { control, target } => {
                let (c, t) = (1u32 << control, 1u32 << target);
                self.perm2(re, im, dim, *control, *target, c, c | t)
            }
            CircuitOp::AntiCNOT { control, target } => {
                self.perm2(re, im, dim, *control, *target, 0, 1u32 << target)
            }
            CircuitOp::SWAP { q1, q2 } => {
                self.perm2(re, im, dim, *q1, *q2, 1u32 << q1, 1u32 << q2)
            }
            CircuitOp::RademacherPhase { .. } => Err("RademacherPhase has no FP32 kernel".into()),
        }
    }

    /// Runs the compiled circuit on the device, updating `state` in place.
    /// On `Err` the state is left untouched so the caller can fall back to the CPU.
    pub fn execute(&self, circuit: &CompiledCircuit, state: &mut QuantumState) -> Result<(), String> {
        if !Self::supports(circuit) {
            return Err("circuit not supported by FP32 GPU backend".into());
        }
        if state.dim != 1usize << circuit.num_qubits {
            return Err("state dimension does not match circuit".into());
        }
        let dim = state.dim;
        let host_re: Vec<f32> = state.real.iter().map(|&x| x as f32).collect();
        let host_im: Vec<f32> = state.imag.iter().map(|&x| x as f32).collect();
        let mut out_re = vec![0f32; dim];
        let mut out_im = vec![0f32; dim];
        unsafe {
            let mut e = 0i32;
            let flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
            let re = (self.d.clCreateBuffer)(self.ctx, flags, dim * 4, host_re.as_ptr() as *mut c_void, &mut e);
            if e != CL_SUCCESS { return Err(format!("alloc failed: {e}")); }
            let im = (self.d.clCreateBuffer)(self.ctx, flags, dim * 4, host_im.as_ptr() as *mut c_void, &mut e);
            if e != CL_SUCCESS {
                (self.d.clReleaseMemObject)(re);
                return Err(format!("alloc failed: {e}"));
            }
            let mut result = Ok(());
            'run: for stage in &circuit.stages {
                for op in &stage.ops {
                    if let Err(m) = self.apply(&re, &im, dim, op) {
                        result = Err(m);
                        break 'run;
                    }
                }
            }
            if result.is_ok() {
                let r1 = (self.d.clEnqueueReadBuffer)(self.queue, re, CL_TRUE, 0, dim * 4, out_re.as_mut_ptr() as *mut c_void, 0, ptr::null(), ptr::null_mut());
                let r2 = (self.d.clEnqueueReadBuffer)(self.queue, im, CL_TRUE, 0, dim * 4, out_im.as_mut_ptr() as *mut c_void, 0, ptr::null(), ptr::null_mut());
                if r1 != CL_SUCCESS || r2 != CL_SUCCESS {
                    result = Err(format!("readback failed: {r1}/{r2}"));
                }
            } else {
                (self.d.clFinish)(self.queue);
            }
            (self.d.clReleaseMemObject)(re);
            (self.d.clReleaseMemObject)(im);
            result?;
        }
        for (dst, src) in state.real.iter_mut().zip(&out_re) { *dst = f64::from(*src); }
        for (dst, src) in state.imag.iter_mut().zip(&out_im) { *dst = f64::from(*src); }
        Ok(())
    }
}

impl Drop for Fp32GpuEngine {
    fn drop(&mut self) {
        unsafe {
            (self.d.clReleaseKernel)(self.k_u1q);
            (self.d.clReleaseKernel)(self.k_perm2);
            (self.d.clReleaseKernel)(self.k_cphase);
            (self.d.clReleaseProgram)(self.program);
            (self.d.clReleaseCommandQueue)(self.queue);
            (self.d.clReleaseContext)(self.ctx);
        }
    }
}
