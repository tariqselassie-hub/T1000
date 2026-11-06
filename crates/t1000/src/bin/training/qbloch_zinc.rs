use t1000::qgnn::graph::MolecularGraph;
use t1000::quaternion::Quaternion;

const WD: f64 = 0.001;
const LR: f64 = 0.01;
const EPS: f64 = 1e-5;

#[derive(Clone, Copy)]
struct Model { w_bond: [Quaternion; 4], w_self: Quaternion, w_read: Quaternion, b_read: Quaternion, w_proj: [f64; 16] }

impl Model {
    fn new() -> Self {
        Self {
            w_bond: [
                Quaternion::new(0.9,0.1,0.1,0.1).normalize(),
                Quaternion::new(0.8,0.3,0.2,0.1).normalize(),
                Quaternion::new(0.7,0.4,0.3,0.2).normalize(),
                Quaternion::new(0.85,0.2,0.25,0.15).normalize(),
            ],
            w_self: Quaternion::new(0.9,0.1,0.05,0.05).normalize(),
            w_read: Quaternion::new(0.5,0.3,0.2,0.1).normalize(),
            b_read: Quaternion::ZERO,
            w_proj: [0.1; 16],
        }
    }
    fn normalize_weights(&mut self) {
        for w in &mut self.w_bond { *w = w.normalize(); }
        self.w_self = self.w_self.normalize();
        self.w_read = self.w_read.normalize();
    }
    fn params(&self) -> Vec<f64> {
        let mut p = Vec::new();
        for w in &self.w_bond { p.extend([w.w,w.x,w.y,w.z]); }
        p.extend([self.w_self.w,self.w_self.x,self.w_self.y,self.w_self.z]);
        p.extend([self.w_read.w,self.w_read.x,self.w_read.y,self.w_read.z]);
        p.extend([self.b_read.w,self.b_read.x,self.b_read.y,self.b_read.z]);
        for v in &self.w_proj { p.push(*v); }
        p
    }
    fn from_params(p: &[f64]) -> Self {
        let mut i = 0;
        let mut wb = [Quaternion::IDENTITY; 4];
        for k in 0..4 { wb[k] = Quaternion::new(p[i],p[i+1],p[i+2],p[i+3]).normalize(); i+=4; }
        let ws = Quaternion::new(p[i],p[i+1],p[i+2],p[i+3]).normalize(); i+=4;
        let wr = Quaternion::new(p[i],p[i+1],p[i+2],p[i+3]).normalize(); i+=4;
        let br = Quaternion::new(p[i],p[i+1],p[i+2],p[i+3]); i+=4;
        let mut wp = [0.0; 16];
                for k in 0..16 { wp[k] = p[i+k]; }
        Self { w_bond: wb, w_self: ws, w_read: wr, b_read: br, w_proj: wp }
    }
    fn forward(&self, g: &MolecularGraph) -> f64 {
        let (pred, _, _, _) = self.forward_full(g);
        pred
    }
    fn forward_full(&self, g: &MolecularGraph) -> (f64, Vec<Quaternion>, Vec<Quaternion>, f64) {
        let h: Vec<Quaternion> = g.node_features.iter().map(|f| {
            let mut q = Quaternion::ZERO;
            for c in 0..4 {
                            let mut val = 0.0;
                            for k in 0..4 {
                                val += self.w_proj[c*4 + k] * f[k];
                            }
                match c { 0=>q.w=val, 1=>q.x=val, 2=>q.y=val, _=>q.z=val }
            }
            q
        }).collect();
        let mut h1 = vec![Quaternion::ZERO; g.num_nodes];
        let mut combineds = vec![Quaternion::ZERO; g.num_nodes];
        for i in 0..g.num_nodes {
            let sr = self.w_self.mul(h[i]).mul(self.w_self.conjugate());
            let mut msgs: Vec<(usize, Quaternion)> = Vec::new();
            for e in &g.edges {
                let j = if e.u == i { e.v } else if e.v == i { e.u } else { continue; };
                let bc = e.bond_class();
                let w = self.w_bond[bc];
                let msg = w.mul(h[j]).mul(w.conjugate());
                msgs.push((j, msg));
            }
            let scores: Vec<f64> = msgs.iter().map(|(_, msg)| {
                h[i].w*msg.w + h[i].x*msg.x + h[i].y*msg.y + h[i].z*msg.z
            }).collect();
            let temp = 0.5;
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| ((s - max_s) / temp).exp()).collect();
            let sum_exp: f64 = exps.iter().sum::<f64>().max(1e-12);
            let mut agg = Quaternion::ZERO;
            for (idx, (_, msg)) in msgs.iter().enumerate() {
                let alpha = exps[idx] / sum_exp;
                agg = agg.add(msg.scale(alpha));
            }
            let combined = sr.add(agg);
            combineds[i] = combined;
            // Hopf fibration activation: S3 -> S2 -> tanh -> lift to S3
            let q = combined.normalize();
            let bx = 2.0 * (q.w * q.x + q.y * q.z);
            let by = 2.0 * (q.w * q.y - q.x * q.z);
            let bz = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z;
            // tanh in 3D (preserves sign, bounded)
            let rx = bx.tanh();
            let ry = by.tanh();
            let rz = bz.tanh();
            let _rn = (rx*rx + ry*ry + rz*rz).sqrt().max(1e-12);
            // Lift back: preserve phase via original w, scale imaginary part
            let scale = (1.0 - rx*rx - ry*ry - rz*rz).abs().sqrt().max(1e-6);
            h1[i] = Quaternion::new(scale, rx, ry, rz).normalize();
        }
        let mut sum_z = 0.0;
        for q in &h1 { sum_z += q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z; }
        let avg_z = sum_z / h1.len().max(1) as f64;
        let avgq = Quaternion::new(avg_z, 0.0, 0.0, 0.0);
        let rot = self.w_read.mul(avgq).mul(self.w_read.conjugate());
        let out = rot.add(self.b_read);
        let pred = out.w*out.w + out.x*out.x - out.y*out.y - out.z*out.z;
        (pred, h1, combineds, avg_z)
    }
    fn grad(&self, g: &MolecularGraph, y: f64) -> Vec<f64> {
        // Forward pass storing intermediates
        let h: Vec<Quaternion> = g.node_features.iter().map(|f| {
            let mut q = Quaternion::ZERO;
            for c in 0..4 {
                            let mut val = 0.0;
                            for k in 0..4 {
                                val += self.w_proj[c*4 + k] * f[k];
                            }
                match c { 0=>q.w=val, 1=>q.x=val, 2=>q.y=val, _=>q.z=val }
            }
            q
        }).collect();
        let mut h1 = vec![Quaternion::ZERO; g.num_nodes];
        let mut combineds = vec![Quaternion::ZERO; g.num_nodes];
        let mut sr_all = vec![Quaternion::ZERO; g.num_nodes];
        let mut msgs_all: Vec<Vec<(usize, Quaternion, usize, f64)>> = vec![Vec::new(); g.num_nodes];
        // msgs: (neighbor, message, bond_class, attention_weight)
        
        for i in 0..g.num_nodes {
            let sr = self.w_self.mul(h[i]).mul(self.w_self.conjugate());
            sr_all[i] = sr;
            let mut msgs: Vec<(usize, Quaternion, usize)> = Vec::new();
            for e in &g.edges {
                let j = if e.u == i { e.v } else if e.v == i { e.u } else { continue; };
                let bc = e.bond_class();
                let w = self.w_bond[bc];
                let msg = w.mul(h[j]).mul(w.conjugate());
                msgs.push((j, msg, bc));
            }
            // attention scores
            let scores: Vec<f64> = msgs.iter().map(|(_, msg, _)| {
                h[i].w*msg.w + h[i].x*msg.x + h[i].y*msg.y + h[i].z*msg.z
            }).collect();
            let temp = 0.5;
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| ((s - max_s) / temp).exp()).collect();
            let sum_exp: f64 = exps.iter().sum::<f64>().max(1e-12);
            let alphas: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();
            let mut agg = Quaternion::ZERO;
            let mut msgs_with_alpha: Vec<(usize, Quaternion, usize, f64)> = Vec::new();
            for (idx, (j, msg, bc)) in msgs.iter().enumerate() {
                let alpha = alphas[idx];
                agg = agg.add(msg.scale(alpha));
                msgs_with_alpha.push((*j, *msg, *bc, alpha));
            }
            msgs_all[i] = msgs_with_alpha;
            let combined = sr.add(agg);
            combineds[i] = combined;
            // Hopf activation
            let q = combined.normalize();
            let bx = 2.0 * (q.w * q.x + q.y * q.z);
            let by = 2.0 * (q.w * q.y - q.x * q.z);
            let bz = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z;
            let rx = bx.tanh();
            let ry = by.tanh();
            let rz = bz.tanh();
            let scale = (1.0 - rx*rx - ry*ry - rz*rz).abs().sqrt().max(1e-6);
            h1[i] = Quaternion::new(scale, rx, ry, rz).normalize();
        }
        
        let n = g.num_nodes as f64;
        let mut sum_z = 0.0;
        for q in &h1 { sum_z += q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z; }
        let avg_z = sum_z / n;
        let avgq = Quaternion::new(avg_z, 0.0, 0.0, 0.0);
        let rot = self.w_read.mul(avgq).mul(self.w_read.conjugate());
        let out = rot.add(self.b_read);
        let pred = out.w*out.w + out.x*out.x - out.y*out.y - out.z*out.z;
        
        // Backward pass
        let dl_dpred = 2.0 * (pred - y);
        let dl_dout = Quaternion::new(
            dl_dpred * 2.0 * out.w,
            dl_dpred * 2.0 * out.x,
            dl_dpred * (-2.0) * out.y,
            dl_dpred * (-2.0) * out.z,
        );
        
        // dL/db_read
        let grad_b_read = dl_dout;
        
        // dL/dw_read (analytic conjugation derivative)
        let mut grad_w_read = Quaternion::ZERO;
        let bases = [Quaternion::IDENTITY,
                     Quaternion::new(0.0, 1.0, 0.0, 0.0),
                     Quaternion::new(0.0, 0.0, 1.0, 0.0),
                     Quaternion::new(0.0, 0.0, 0.0, 1.0)];
        for ci in 0..4 {
            let e = bases[ci];
            let e_star = e.conjugate();
            let dr = e.mul(avgq).mul(self.w_read.conjugate())
                .add(self.w_read.mul(avgq).mul(e_star));
            let out = self.w_read.mul(avgq).mul(self.w_read.conjugate()).add(self.b_read);
            let dpred = 2.0 * out.w * dr.w + 2.0 * out.x * dr.x
                - 2.0 * out.y * dr.y - 2.0 * out.z * dr.z;
            let g = dl_dpred * dpred;
            match ci { 0 => grad_w_read.w = g, 1 => grad_w_read.x = g,
                       2 => grad_w_read.y = g, _ => grad_w_read.z = g }
        }
        
        // dL/davg_z
        let e0 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let col0 = self.w_read.mul(e0).mul(self.w_read.conjugate());
        let dl_davg_z = dl_dout.w*col0.w + dl_dout.x*col0.x + dl_dout.y*col0.y + dl_dout.z*col0.z;
        
        // dL/dh1[i] = (dl_davg_z / n) * (2*h1.w, 2*h1.x, -2*h1.y, -2*h1.z)
        // dL/dcombined[i] through Hopf activation (analytic via finite diff on Hopf function)
        let mut dl_dcombined = vec![Quaternion::ZERO; g.num_nodes];
        for i in 0..g.num_nodes {
            let hi = h1[i];
            let bloch_deriv = Quaternion::new(2.0*hi.w, 2.0*hi.x, -2.0*hi.y, -2.0*hi.z);
            let dl_dh = Quaternion::new(dl_davg_z / n * bloch_deriv.w, dl_davg_z / n * bloch_deriv.x, dl_davg_z / n * bloch_deriv.y, dl_davg_z / n * bloch_deriv.z);
            
            // Analytic Hopf gradient: finite diff on the Hopf function only (4 evals)
            let combined = combineds[i];
            let _grad_combined = Quaternion::ZERO;
            for ci in 0..4 {
                let mut cp = combined; let mut cm = combined;
                match ci { 0=>{cp.w+=EPS; cm.w-=EPS;}, 1=>{cp.x+=EPS; cm.x-=EPS;},
                           2=>{cp.y+=EPS; cm.y-=EPS;}, _=>{cp.z+=EPS; cm.z-=EPS;} }
                // Hopf forward for cp
                let qp = cp.normalize();
                let bxp = 2.0*(qp.w*qp.x+qp.y*qp.z);
                let byp = 2.0*(qp.w*qp.y-qp.x*qp.z);
                let bzp = qp.w*qp.w+qp.x*qp.x-qp.y*qp.y-qp.z*qp.z;
                let rxp = bxp.tanh(); let ryp = byp.tanh(); let rzp = bzp.tanh();
                let scalep = (1.0-rxp*rxp-ryp*ryp-rzp*rzp).abs().sqrt().max(1e-6);
                let hp = Quaternion::new(scalep, rxp, ryp, rzp).normalize();
                
                let qm = cm.normalize();
                let bxm = 2.0*(qm.w*qm.x+qm.y*qm.z);
                let bym = 2.0*(qm.w*qm.y-qm.x*qm.z);
                let bzm = qm.w*qm.w+qm.x*qm.x-qm.y*qm.y-qm.z*qm.z;
                let rxm = bxm.tanh(); let rym = bym.tanh(); let rzm = bzm.tanh();
                let scalem = (1.0-rxm*rxm-rym*rym-rzm*rzm).abs().sqrt().max(1e-6);
                let hm = Quaternion::new(scalem, rxm, rym, rzm).normalize();
                
                let dh = Quaternion::new((hp.w-hm.w)/(2.0*EPS), (hp.x-hm.x)/(2.0*EPS), (hp.y-hm.y)/(2.0*EPS), (hp.z-hm.z)/(2.0*EPS));
                let contrib = dh.w*dl_dh.w + dh.x*dl_dh.x + dh.y*dl_dh.y + dh.z*dl_dh.z;
                let mut g = dl_dcombined[i];
                match ci { 0=>g.w=contrib, 1=>g.x=contrib, 2=>g.y=contrib, _=>g.z=contrib }
                dl_dcombined[i] = g;
            }
        }
        
        // Analytic full-graph backprop: compute dl_dh (gradient wrt h_i)
        let tau = 0.5;
        let mut dl_dh = vec![Quaternion::ZERO; g.num_nodes];
        for i in 0..g.num_nodes {
            dl_dh[i] = self.w_self.conjugate().mul(dl_dcombined[i]).mul(self.w_self);
        }
        for i in 0..g.num_nodes {
            let mut bar_m = Quaternion::ZERO;
            for (_j, msg, _bc, alpha) in &msgs_all[i] {
                bar_m.w += alpha * msg.w; bar_m.x += alpha * msg.x;
                bar_m.y += alpha * msg.y; bar_m.z += alpha * msg.z;
            }
            for (j, msg, bc, alpha) in &msgs_all[i] {
                let w_bond = self.w_bond[*bc];
                let w_bond_star = w_bond.conjugate();
                let gC = dl_dcombined[i];
                let dot_g_msg = gC.w*msg.w + gC.x*msg.x + gC.y*msg.y + gC.z*msg.z;
                let conj_g = w_bond_star.mul(gC).mul(w_bond);
                let alpha = *alpha;
                dl_dh[*j].w += alpha * conj_g.w;
                dl_dh[*j].x += alpha * conj_g.x;
                dl_dh[*j].y += alpha * conj_g.y;
                dl_dh[*j].z += alpha * conj_g.z;
                let h_i = h[i];
                let conj_h = w_bond_star.mul(h_i).mul(w_bond);
                let scale = dot_g_msg * alpha * (1.0 - alpha) / tau;
                dl_dh[*j].w += scale * conj_h.w;
                dl_dh[*j].x += scale * conj_h.x;
                dl_dh[*j].y += scale * conj_h.y;
                dl_dh[*j].z += scale * conj_h.z;
                let diff_w = msg.w - bar_m.w;
                let diff_x = msg.x - bar_m.x;
                let diff_y = msg.y - bar_m.y;
                let diff_z = msg.z - bar_m.z;
                let scale2 = dot_g_msg * alpha / tau;
                dl_dh[i].w += scale2 * diff_w;
                dl_dh[i].x += scale2 * diff_x;
                dl_dh[i].y += scale2 * diff_y;
                dl_dh[i].z += scale2 * diff_z;
            }
        }
        let mut grad_w_proj = [0.0; 16];
        for i in 0..g.num_nodes {
            let x = &g.node_features[i];
            let gz = dl_dh[i];
            let comps = [gz.w, gz.x, gz.y, gz.z];
            for c in 0..4 {
                for k in 0..4 {
                    grad_w_proj[c*4 + k] += comps[c] * x[k];
                }
            }
        }
        // Weight gradients via finite differences (weights are few: 28 params)
        let mut grad_w_self = Quaternion::ZERO;
        let mut grad_w_bond = [Quaternion::ZERO; 4];
        let bases = [Quaternion::IDENTITY,
                     Quaternion::new(0.0, 1.0, 0.0, 0.0),
                     Quaternion::new(0.0, 0.0, 1.0, 0.0),
                     Quaternion::new(0.0, 0.0, 0.0, 1.0)];
        
        // w_self gradient
        for ci in 0..4 {
            // dL/dw_self = sum_i dL/dcombined[i] · d(sr_i)/dw_self
            // sr_i = w_self * h[i] * w_self^*
            let mut total = 0.0;
            for i in 0..g.num_nodes {
                let e = bases[ci];
                let e_star = e.conjugate();
                let dr = e.mul(h[i]).mul(self.w_self.conjugate()).add(self.w_self.mul(h[i]).mul(e_star));
                total += dr.w*dl_dcombined[i].w + dr.x*dl_dcombined[i].x + dr.y*dl_dcombined[i].y + dr.z*dl_dcombined[i].z;
            }
            match ci { 0=>grad_w_self.w=total, 1=>grad_w_self.x=total,
                       2=>grad_w_self.y=total, _=>grad_w_self.z=total }
        }
        
        // w_bond gradient (per bond class)
        for bc in 0..4 {
            for ci in 0..4 {
                let mut total = 0.0;
                for i in 0..g.num_nodes {
                    for (j, _msg, msg_bc, alpha) in &msgs_all[i] {
                        if *msg_bc != bc { continue; }
                        // d(msg)/dw_bond[bc]
                        let h_j = h[*j];
                        let e = bases[ci];
                        let e_star = e.conjugate();
                        let dmsg = e.mul(h_j).mul(self.w_bond[bc].conjugate()).add(self.w_bond[bc].mul(h_j).mul(e_star));
                        total += dmsg.w*dl_dcombined[i].w*alpha + dmsg.x*dl_dcombined[i].x*alpha + dmsg.y*dl_dcombined[i].y*alpha + dmsg.z*dl_dcombined[i].z*alpha;
                    }
                }
                let gw = &mut grad_w_bond[bc];
                match ci { 0=>gw.w=total, 1=>gw.x=total, 2=>gw.y=total, _=>gw.z=total }
            }
        }
        
        
        // Assemble gradient vector (add w_proj at end)
        let mut grads = Vec::new();
        for w in &grad_w_bond { grads.extend([w.w, w.x, w.y, w.z]); }
        grads.extend([grad_w_self.w, grad_w_self.x, grad_w_self.y, grad_w_self.z]);
        grads.extend([grad_w_read.w, grad_w_read.x, grad_w_read.y, grad_w_read.z]);
        grads.extend([grad_b_read.w, grad_b_read.x, grad_b_read.y, grad_b_read.z]);
        for v in &grad_w_proj { grads.push(*v); }
        let base = self.params();
        for pi in 0..grads.len() { grads[pi] += WD * base[pi]; }
        grads
    }
}

fn mse(model: &Model, data: &[(MolecularGraph, f64)]) -> f64 {
    let mut s = 0.0;
    for (g, y) in data { let p = model.forward(g); s += (p - y).powi(2); }
    s / data.len() as f64
}

fn main() {
    // Load ZINC CSV
    let path = "C:/Users/zoddj/datasets/zinc_250k.zip";
    // We'll read extracted csv from a known path; unzip manually not here. Instead read first 5000 from extracted file if present.
    let csv_path = "C:/Users/zoddj/datasets/combined/zinc_qm9_10k.csv";
    let text = std::fs::read_to_string(csv_path).expect("csv");
    let mut data: Vec<(MolecularGraph, f64)> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 { continue; }
        if data.len() >= 5000 { break; }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 7 { continue; }
        if cols[5] != "1" { continue; }
        let smiles = cols[0].to_string();
        let target: f64 = match cols[1].parse() { Ok(v) => v, Err(_) => continue };
        if let Ok(g) = MolecularGraph::from_smiles(&smiles, 12) {
            data.push((g, target));
        }
    }
    println!("usable: {}", data.len());

    let n = data.len() as f64;
    let mean: f64 = data.iter().map(|d| d.1).sum::<f64>() / n;
    let var: f64 = data.iter().map(|d| (d.1-mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-9);
    for d in data.iter_mut() { d.1 = (d.1 - mean) / std; }

    let split = (data.len() as f64 * 0.8) as usize;
    let (train, test) = data.split_at(split);

    let mut model = Model::new();
    for epoch in 0..20 {
        let mut total = 0.0;
        for (g, y) in train {
            let pred = model.forward(g);
            total += (pred - y).powi(2);
            let grads = model.grad(g, *y);
            let base = model.params();
            let mut newp = base.clone();
            for pi in 0..base.len() { newp[pi] -= LR * grads[pi]; }
            model = Model::from_params(&newp);
            model.normalize_weights();
        }
        println!("epoch {} train MSE = {:.4}", epoch, total / train.len() as f64);
    }

    let test_mse = mse(&model, test);
    let test_mae: f64 = test.iter().map(|(g,y)| (model.forward(g)-y).abs()).sum::<f64>() / test.len() as f64;
    let baseline_mae: f64 = test.iter().map(|(_,y)| y.abs()).sum::<f64>() / test.len() as f64;
    println!("test MSE = {:.4}, MAE = {:.4}", test_mse, test_mae);
    println!("baseline (predict mean) MAE = {:.4}", baseline_mae);
}
