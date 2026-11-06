// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

pub type Quat = [f64; 4];
pub const FEAT_DIM: usize = 19;
pub const MP_LAYERS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetSource {
    QM9,
    Zinc,
}

#[derive(Clone, Debug)]
pub struct MolGraph {
    pub features: Vec<[f64; FEAT_DIM]>,
    pub adj: Vec<Vec<usize>>,
}

impl MolGraph {
    #[inline]
    pub fn num_atoms(&self) -> usize {
        self.features.len()
    }
}

fn quat_mul(a: Quat, b: Quat) -> Quat {
    [
        a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
        a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
        a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
        a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
    ]
}
fn quat_conj(q: Quat) -> Quat { [q[0], -q[1], -q[2], -q[3]] }
fn quat_dot(a: Quat, b: Quat) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3] }
fn quat_norm(q: Quat) -> f64 { quat_dot(q,q).sqrt().max(1e-12) }
fn normalize_quat(q: Quat) -> Quat { let inv=1.0/quat_norm(q); [q[0]*inv,q[1]*inv,q[2]*inv,q[3]*inv] }
fn add_quat(a:Quat,b:Quat)->Quat{[a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]]}
fn scale_quat(q:Quat,s:f64)->Quat{[q[0]*s,q[1]*s,q[2]*s,q[3]*s]}
fn quat_conjugate_rot(w:Quat,q:Quat)->Quat{quat_mul(quat_mul(w,q),quat_conj(w))}
fn unit_lift_backward(q:Quat,norm_z:f64,g:Quat)->Quat{
    let proj=quat_dot(g,q); let inv=1.0/norm_z.max(1e-12);
    [(g[0]-proj*q[0])*inv,(g[1]-proj*q[1])*inv,(g[2]-proj*q[2])*inv,(g[3]-proj*q[3])*inv]
}
fn quat_conjugate_backward(w:Quat,q:Quat,dy:Quat)->(Quat,Quat){
    let w_conj=quat_conj(w); let wq=quat_mul(w,q);
    let (d_wq,d_w_conj)=quat_mul_backward(wq,w_conj,dy);
    let (d_w_left,d_q)=quat_mul_backward(w,q,d_wq);
    let d_w_right=quat_conj(d_w_conj);
    (add_quat(d_w_left,d_w_right),d_q)
}
fn quat_mul_backward(p:Quat,q:Quat,dy:Quat)->(Quat,Quat){
    (quat_mul(dy,quat_conj(q)), quat_mul(quat_conj(p),dy))
}

#[derive(Clone, Debug)]
pub struct LayerTape {
    pub q_in: Vec<Quat>,
    pub m_bond: Vec<Quat>,
    pub attn_weights: Vec<Vec<f64>>,
    pub u_norm: Vec<f64>,
    pub q_out: Vec<Quat>,
}
#[derive(Clone, Debug)]
pub struct ForwardTape {
    pub z0_norm: Vec<f64>,
    pub layers: Vec<LayerTape>,
}
#[derive(Clone, Debug)]
pub struct MultiTaskBackboneGrads {
    pub w_proj: [[f64; FEAT_DIM]; 4],
    pub w_self_qm9: [Quat; MP_LAYERS],
    pub w_bond_qm9: [Quat; MP_LAYERS],
    pub w_self_zinc: [Quat; MP_LAYERS],
    pub w_bond_zinc: [Quat; MP_LAYERS],
}
impl MultiTaskBackboneGrads {
    pub fn zeros() -> Self {
        Self {
            w_proj: [[0.0; FEAT_DIM]; 4],
            w_self_qm9: [[0.0; 4]; MP_LAYERS],
            w_bond_qm9: [[0.0; 4]; MP_LAYERS],
            w_self_zinc: [[0.0; 4]; MP_LAYERS],
            w_bond_zinc: [[0.0; 4]; MP_LAYERS],
        }
    }
}
#[derive(Clone, Debug)]
pub struct MultiTaskBackbone {
    pub w_proj: [[f64; FEAT_DIM]; 4],
    pub w_self_qm9: [Quat; MP_LAYERS],
    pub w_bond_qm9: [Quat; MP_LAYERS],
    pub w_self_zinc: [Quat; MP_LAYERS],
    pub w_bond_zinc: [Quat; MP_LAYERS],
    pub attn_temp: f64,
    pub residual_beta: f64,
}
impl MultiTaskBackbone {
    pub fn new(seed: u64, attn_temp: f64, residual_beta: f64) -> Self {
        let mut rng = seed.max(1);
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            ((rng as f64)/(u64::MAX as f64))*2.0 - 1.0
        };
        let mut w_proj = [[0.0; FEAT_DIM]; 4];
        let scale = (2.0/(FEAT_DIM as f64)).sqrt();
        for c in 0..4 { for f in 0..FEAT_DIM { w_proj[c][f]=next_f64()*scale; } }
        let mut init_rotors = || {
            let mut rotors = [[0.0;4]; MP_LAYERS];
            for l in 0..MP_LAYERS {
                rotors[l]=normalize_quat([1.0+0.1*next_f64(),0.1*next_f64(),0.1*next_f64(),0.1*next_f64()]);
            }
            rotors
        };
        Self { w_proj, w_self_qm9: init_rotors(), w_bond_qm9: init_rotors(), w_self_zinc: init_rotors(), w_bond_zinc: init_rotors(), attn_temp, residual_beta }
    }
    #[inline]
    pub fn num_layers(&self) -> usize { MP_LAYERS }
    pub fn forward(&self, graph: &MolGraph, source: DatasetSource, edge_dropout_rate: f64, sample_seed: u64) -> (Vec<Quat>, ForwardTape) {
        let n = graph.num_atoms();
        let mut rng = sample_seed.wrapping_add(0x9E37_79B9_7F4A_7C15).max(1);
        let mut next_unit = || -> f64 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng as f64)/(u64::MAX as f64) };
        let (w_self_ref, w_bond_ref) = match source { DatasetSource::QM9 => (&self.w_self_qm9, &self.w_bond_qm9), DatasetSource::Zinc => (&self.w_self_zinc, &self.w_bond_zinc) };
        let mut z0_norm = Vec::with_capacity(n);
        let mut q_curr = Vec::with_capacity(n);
        for i in 0..n {
            let x=&graph.features[i];
            let mut z=[0.0_f64;4];
            for c in 0..4 { let mut sum=0.0; for f in 0..FEAT_DIM { sum += self.w_proj[c][f]*x[f]; } z[c]=sum; }
            let norm=quat_norm(z);
            z0_norm.push(norm);
            q_curr.push([z[0]/norm,z[1]/norm,z[2]/norm,z[3]/norm]);
        }
        let mut layers_tape = Vec::with_capacity(MP_LAYERS);
        let inv_temp = 1.0/self.attn_temp;
        for l in 0..MP_LAYERS {
            let w_s = w_self_ref[l]; let w_b = w_bond_ref[l];
            let m_bond: Vec<Quat> = q_curr.iter().map(|&qj| quat_conjugate_rot(w_b,qj)).collect();
            let mut attn_weights = Vec::with_capacity(n);
            let mut u_norm = Vec::with_capacity(n);
            let mut q_next = Vec::with_capacity(n);
            for i in 0..n {
                let qi=q_curr[i];
                let self_rot=quat_conjugate_rot(w_s,qi);
                let neighbors=&graph.adj[i];
                let mut agg_msg=[0.0_f64;4];
                let mut alphas=vec![0.0_f64;neighbors.len()];
                if !neighbors.is_empty() {
                    let mut scores=vec![f64::NEG_INFINITY;neighbors.len()];
                    let mut max_score=f64::NEG_INFINITY;
                    let mut active_edges=0usize;
                    for (idx,&j) in neighbors.iter().enumerate() {
                        let drop_edge = edge_dropout_rate>0.0 && neighbors.len()>1 && next_unit()<edge_dropout_rate;
                        if !drop_edge {
                            let s=quat_dot(qi,q_curr[j])*inv_temp;
                            scores[idx]=s;
                            if s>max_score { max_score=s; }
                            active_edges+=1;
                        }
                    }
                    if active_edges==0 {
                        let j0=neighbors[0];
                        let s=quat_dot(qi,q_curr[j0])*inv_temp;
                        scores[0]=s; max_score=s;
                    }
                    let mut denom=0.0_f64;
                    for (idx,&s) in scores.iter().enumerate() {
                        if s.is_finite() { let e=(s-max_score).exp(); alphas[idx]=e; denom+=e; }
                    }
                    let inv_denom=1.0/denom.max(1e-12);
                    for (idx,&j) in neighbors.iter().enumerate() {
                        alphas[idx]*=inv_denom;
                        let a=alphas[idx];
                        if a>0.0 {
                            let mj=m_bond[j];
                            agg_msg[0]+=a*mj[0]; agg_msg[1]+=a*mj[1]; agg_msg[2]+=a*mj[2]; agg_msg[3]+=a*mj[3];
                        }
                    }
                }
                let res_qi=scale_quat(qi,self.residual_beta);
                let u_i=add_quat(add_quat(self_rot,agg_msg),res_qi);
                let norm_u=quat_norm(u_i);
                let q_out_i=[u_i[0]/norm_u,u_i[1]/norm_u,u_i[2]/norm_u,u_i[3]/norm_u];
                attn_weights.push(alphas); u_norm.push(norm_u); q_next.push(q_out_i);
            }
            layers_tape.push(LayerTape{q_in:q_curr,m_bond,attn_weights,u_norm,q_out:q_next.clone()});
            q_curr=q_next;
        }
        (q_curr, ForwardTape{z0_norm,layers:layers_tape})
    }
    pub fn backward(&self, graph:&MolGraph, source:DatasetSource, tape:&ForwardTape, d_q_final:&[Quat], grads:&mut MultiTaskBackboneGrads) {
        let n=graph.num_atoms();
        let inv_temp=1.0/self.attn_temp;
        let mut d_q_curr: Vec<Quat>=d_q_final.to_vec();
        let (w_self_ref,w_bond_ref)=match source { DatasetSource::QM9=>(&self.w_self_qm9,&self.w_bond_qm9), DatasetSource::Zinc=>(&self.w_self_zinc,&self.w_bond_zinc) };
        let mut task_w_self_grads=[[0.0;4];MP_LAYERS];
        let mut task_w_bond_grads=[[0.0;4];MP_LAYERS];
        for l in (0..MP_LAYERS).rev() {
            let lt=&tape.layers[l];
            let w_s=w_self_ref[l]; let w_b=w_bond_ref[l];
            let mut d_q_prev=vec![[0.0_f64;4];n];
            let mut d_m_bond=vec![[0.0_f64;4];n];
            for i in 0..n {
                let d_u_i=unit_lift_backward(lt.q_out[i],lt.u_norm[i],d_q_curr[i]);
                d_q_prev[i]=add_quat(d_q_prev[i],scale_quat(d_u_i,self.residual_beta));
                let (dw_s,dq_self)=quat_conjugate_backward(w_s,lt.q_in[i],d_u_i);
                task_w_self_grads[l]=add_quat(task_w_self_grads[l],dw_s);
                d_q_prev[i]=add_quat(d_q_prev[i],dq_self);
                let neighbors=&graph.adj[i];
                let alphas=&lt.attn_weights[i];
                if !neighbors.is_empty() {
                    let mut dot_du_m=Vec::with_capacity(neighbors.len());
                    let mut expected_dot=0.0_f64;
                    for (idx,&j) in neighbors.iter().enumerate() {
                        let a=alphas[idx];
                        let mj=lt.m_bond[j];
                        let d_alpha=quat_dot(d_u_i,mj);
                        dot_du_m.push(d_alpha);
                        expected_dot+=a*d_alpha;
                        if a>0.0 { d_m_bond[j]=add_quat(d_m_bond[j],scale_quat(d_u_i,a)); }
                    }
                    for (idx,&j) in neighbors.iter().enumerate() {
                        let a=alphas[idx];
                        if a>0.0 {
                            let d_score=a*(dot_du_m[idx]-expected_dot)*inv_temp;
                            d_q_prev[i]=add_quat(d_q_prev[i],scale_quat(lt.q_in[j],d_score));
                            d_q_prev[j]=add_quat(d_q_prev[j],scale_quat(lt.q_in[i],d_score));
                        }
                    }
                }
            }
            for j in 0..n {
                let (dw_b,dq_bond)=quat_conjugate_backward(w_b,lt.q_in[j],d_m_bond[j]);
                task_w_bond_grads[l]=add_quat(task_w_bond_grads[l],dw_b);
                d_q_prev[j]=add_quat(d_q_prev[j],dq_bond);
            }
            d_q_curr=d_q_prev;
        }
        match source {
            DatasetSource::QM9 => {
                for l in 0..MP_LAYERS {
                    grads.w_self_qm9[l]=add_quat(grads.w_self_qm9[l],task_w_self_grads[l]);
                    grads.w_bond_qm9[l]=add_quat(grads.w_bond_qm9[l],task_w_bond_grads[l]);
                }
            }
            DatasetSource::Zinc => {
                for l in 0..MP_LAYERS {
                    grads.w_self_zinc[l]=add_quat(grads.w_self_zinc[l],task_w_self_grads[l]);
                    grads.w_bond_zinc[l]=add_quat(grads.w_bond_zinc[l],task_w_bond_grads[l]);
                }
            }
        }
        let q0_slice=&tape.layers[0].q_in;
        for i in 0..n {
            let d_z0_i=unit_lift_backward(q0_slice[i],tape.z0_norm[i],d_q_curr[i]);
            let x_i=&graph.features[i];
            for c in 0..4 {
                let gz=d_z0_i[c];
                for f in 0..FEAT_DIM { grads.w_proj[c][f]+=gz*x_i[f]; }
            }
        }
    }
    pub fn step_normalized_sgd(&mut self, grads:&MultiTaskBackboneGrads, inv_b:f64, lr:f64, wd:f64) {
        for c in 0..4 { for f in 0..FEAT_DIM { self.w_proj[c][f] -= lr*(grads.w_proj[c][f]*inv_b + wd*self.w_proj[c][f]); } }
        for l in 0..MP_LAYERS {
            for c in 0..4 {
                self.w_self_qm9[l][c] -= lr*(grads.w_self_qm9[l][c]*inv_b);
                self.w_bond_qm9[l][c] -= lr*(grads.w_bond_qm9[l][c]*inv_b);
                self.w_self_zinc[l][c] -= lr*(grads.w_self_zinc[l][c]*inv_b);
                self.w_bond_zinc[l][c] -= lr*(grads.w_bond_zinc[l][c]*inv_b);
            }
            self.w_self_qm9[l]=normalize_quat(self.w_self_qm9[l]);
            self.w_bond_qm9[l]=normalize_quat(self.w_bond_qm9[l]);
            self.w_self_zinc[l]=normalize_quat(self.w_self_zinc[l]);
            self.w_bond_zinc[l]=normalize_quat(self.w_bond_zinc[l]);
        }
    }
}
