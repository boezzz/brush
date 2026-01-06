use std::io;
use tokio::io::{AsyncRead, AsyncReadExt};

// magic number
const MAGIC_DFSS: &[u8; 4] = b"DFSS";


#[derive(Clone, Debug)]
pub struct FrameDeformation {
    pub positions: Vec<f32>,
    pub rotations: Vec<f32>,
    pub scales: Vec<f32>,
    pub is_absolute: bool,
}

#[derive(Clone, Debug)]
pub struct Deformations {
    pub num_frames: u32,
    pub num_points: u32,
    pub has_scales: bool,
    positions: Vec<f32>,
    rotations: Vec<f32>,
    scales: Vec<f32>,
}

pub struct AllFrameDeformations {
    pub all_means: Vec<f32>,
    pub all_rotations: Vec<f32>,
    pub all_log_scales: Vec<f32>,
    pub num_frames: u32,
    pub num_splats: usize,
}

pub struct SeparatedFrameDeformations {
    pub dynamic_means: Vec<f32>,
    pub dynamic_rotations: Vec<f32>,
    pub dynamic_log_scales: Vec<f32>,
    pub static_means: Vec<f32>,
    pub static_rotations: Vec<f32>,
    pub static_log_scales: Vec<f32>,
    pub num_frames: u32,
    pub num_deformable: usize,
    pub num_static: usize,
}

impl Deformations {
    pub fn get_frame(&self, frame: u32) -> Option<FrameDeformation> {
        if frame >= self.num_frames {
            return None;
        }

        let n = self.num_points as usize;
        let frame = frame as usize;

        let pos_start = frame * n * 3;
        let pos_end = pos_start + n * 3;

        let rot_start = frame * n * 4;
        let rot_end = rot_start + n * 4;

        let scales = if self.has_scales {
            let scale_start = frame * n * 3;
            let scale_end = scale_start + n * 3;
            self.scales[scale_start..scale_end].to_vec()
        } else {
            vec![]
        };

        Some(FrameDeformation {
            positions: self.positions[pos_start..pos_end].to_vec(),
            rotations: self.rotations[rot_start..rot_end].to_vec(),
            scales,
            is_absolute: self.has_scales,
        })
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "deformation file too small",
            ));
        }

        let magic = &data[0..4];
        if magic != MAGIC_DFSS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported deformation format: {:?}",
                    magic
                ),
            ));
        }
        let has_scales = true;

        let num_frames = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let num_points = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

        let n = num_points as usize;
        let t = num_frames as usize;

        let pos_size = t * n * 3 * 4;
        let rot_size = t * n * 4 * 4;
        let scale_size = if has_scales { t * n * 3 * 4 } else { 0 };

        let expected_size = 12 + pos_size + rot_size + scale_size;
        // size check
        if data.len() < expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "deformation file size mismatch"
                ),
            ));
        }

        let pos_data = &data[12..12 + pos_size];
        let positions: Vec<f32> = pos_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let rot_data = &data[12 + pos_size..12 + pos_size + rot_size];
        let rotations: Vec<f32> = rot_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let scales = if has_scales {
            let scale_data = &data[12 + pos_size + rot_size..12 + pos_size + rot_size + scale_size];
            scale_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        } else {
            vec![]
        };

        Ok(Deformations {
            num_frames,
            num_points,
            has_scales,
            positions,
            rotations,
            scales,
        })
    }

    pub async fn from_reader<R: AsyncRead + Unpin>(mut reader: R) -> io::Result<Self> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).await?;
        Self::from_bytes(&data)
    }

    pub fn apply_all_frames_separated(
        &self,
        base_means: &[f32],
        base_rotations: &[f32],
        base_log_scales: &[f32],
    ) -> SeparatedFrameDeformations {
        let total_splats = base_means.len() / 3;
        let n_deformable = self.num_points as usize;
        let n_static = total_splats.saturating_sub(n_deformable);
        let num_frames = self.num_frames as usize;

        // only allocate memory for dynamic gaussians
        let mut dynamic_means = Vec::with_capacity(num_frames * n_deformable * 3);
        let mut dynamic_rotations = Vec::with_capacity(num_frames * n_deformable * 4);
        let mut dynamic_log_scales = Vec::with_capacity(num_frames * n_deformable * 3);

        for frame in 0..num_frames {
            let pos_start = frame * n_deformable * 3;
            let rot_start = frame * n_deformable * 4;
            let scale_start = frame * n_deformable * 3;

            for i in 0..n_deformable {
                let idx = pos_start + i * 3;
                dynamic_means.push(self.positions[idx]);
                dynamic_means.push(self.positions[idx + 1]);
                dynamic_means.push(self.positions[idx + 2]);
            }

            for i in 0..n_deformable {
                let idx = rot_start + i * 4;
                dynamic_rotations.push(self.rotations[idx]);
                dynamic_rotations.push(self.rotations[idx + 1]);
                dynamic_rotations.push(self.rotations[idx + 2]);
                dynamic_rotations.push(self.rotations[idx + 3]);
            }

            let n_deformable_scales = if self.has_scales { n_deformable } else { 0 };
            for i in 0..n_deformable_scales {
                let idx = scale_start + i * 3;
                dynamic_log_scales.push(self.scales[idx].abs().max(1e-10).ln());
                dynamic_log_scales.push(self.scales[idx + 1].abs().max(1e-10).ln());
                dynamic_log_scales.push(self.scales[idx + 2].abs().max(1e-10).ln());
            }
        }

        // static gaussians
        let mut static_means = Vec::with_capacity(n_static * 3);
        let mut static_rotations = Vec::with_capacity(n_static * 4);
        let mut static_log_scales = Vec::with_capacity(n_static * 3);

        for i in n_deformable..total_splats {
            let base_idx = i * 3;
            static_means.push(base_means[base_idx]);
            static_means.push(base_means[base_idx + 1]);
            static_means.push(base_means[base_idx + 2]);

            let rot_idx = i * 4;
            static_rotations.push(base_rotations[rot_idx]);
            static_rotations.push(base_rotations[rot_idx + 1]);
            static_rotations.push(base_rotations[rot_idx + 2]);
            static_rotations.push(base_rotations[rot_idx + 3]);

            static_log_scales.push(base_log_scales[base_idx]);
            static_log_scales.push(base_log_scales[base_idx + 1]);
            static_log_scales.push(base_log_scales[base_idx + 2]);
        }

        SeparatedFrameDeformations {
            dynamic_means,
            dynamic_rotations,
            dynamic_log_scales,
            static_means,
            static_rotations,
            static_log_scales,
            num_frames: self.num_frames,
            num_deformable: n_deformable,
            num_static: n_static,
        }
    }

    pub fn apply_all_frames(
        &self,
        base_means: &[f32],
        base_rotations: &[f32],
        base_log_scales: &[f32],
    ) -> AllFrameDeformations {
        let total_splats = base_means.len() / 3;
        let n_deformable = self.num_points as usize;
        let num_frames = self.num_frames as usize;

        let mut all_means = Vec::with_capacity(num_frames * total_splats * 3);
        let mut all_rotations = Vec::with_capacity(num_frames * total_splats * 4);
        let mut all_log_scales = Vec::with_capacity(num_frames * total_splats * 3);

        for frame in 0..num_frames {
            let pos_start = frame * n_deformable * 3;
            let rot_start = frame * n_deformable * 4;
            let scale_start = frame * n_deformable * 3;

            for i in 0..total_splats {
                if i < n_deformable {
                    let idx = pos_start + i * 3;
                    all_means.push(self.positions[idx]);
                    all_means.push(self.positions[idx + 1]);
                    all_means.push(self.positions[idx + 2]);
                } else {
                    let base_idx = i * 3;
                    all_means.push(base_means[base_idx]);
                    all_means.push(base_means[base_idx + 1]);
                    all_means.push(base_means[base_idx + 2]);
                }
            }

            for i in 0..total_splats {
                if i < n_deformable {
                    let idx = rot_start + i * 4;
                    all_rotations.push(self.rotations[idx]);
                    all_rotations.push(self.rotations[idx + 1]);
                    all_rotations.push(self.rotations[idx + 2]);
                    all_rotations.push(self.rotations[idx + 3]);
                } else {
                    let base_idx = i * 4;
                    all_rotations.push(base_rotations[base_idx]);
                    all_rotations.push(base_rotations[base_idx + 1]);
                    all_rotations.push(base_rotations[base_idx + 2]);
                    all_rotations.push(base_rotations[base_idx + 3]);
                }
            }

            let n_deformable_scales = if self.has_scales { n_deformable } else { 0 };
            for i in 0..total_splats {
                if i < n_deformable_scales {
                    let idx = scale_start + i * 3;
                    all_log_scales.push(self.scales[idx].abs().max(1e-10).ln());
                    all_log_scales.push(self.scales[idx + 1].abs().max(1e-10).ln());
                    all_log_scales.push(self.scales[idx + 2].abs().max(1e-10).ln());
                } else {
                    let base_idx = i * 3;
                    all_log_scales.push(base_log_scales[base_idx]);
                    all_log_scales.push(base_log_scales[base_idx + 1]);
                    all_log_scales.push(base_log_scales[base_idx + 2]);
                }
            }
        }

        AllFrameDeformations {
            all_means,
            all_rotations,
            all_log_scales,
            num_frames: self.num_frames,
            num_splats: total_splats,
        }
    }
}

pub fn apply_deformation(
    base_means: &[f32],
    base_rotations: &[f32],
    base_log_scales: &[f32],
    deform: &FrameDeformation,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total_splats = base_means.len() / 3;
    let n_deformable = deform.positions.len() / 3;

    // only DFSS is supported
    apply_absolute_deformation(
        base_means,
        base_rotations,
        base_log_scales,
        deform,
        total_splats,
        n_deformable,
    )
}

fn apply_absolute_deformation(
    base_means: &[f32],
    base_rotations: &[f32],
    base_log_scales: &[f32],
    deform: &FrameDeformation,
    total_splats: usize,
    n_deformable: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut deformed_means = Vec::with_capacity(base_means.len());
    for i in 0..total_splats {
        if i < n_deformable {
            let idx = i * 3;
            deformed_means.push(deform.positions[idx]);
            deformed_means.push(deform.positions[idx + 1]);
            deformed_means.push(deform.positions[idx + 2]);
        } else {
            let base_idx = i * 3;
            deformed_means.push(base_means[base_idx]);
            deformed_means.push(base_means[base_idx + 1]);
            deformed_means.push(base_means[base_idx + 2]);
        }
    }

    let mut deformed_rotations = Vec::with_capacity(base_rotations.len());
    for i in 0..total_splats {
        if i < n_deformable {
            let idx = i * 4;
            deformed_rotations.push(deform.rotations[idx]);
            deformed_rotations.push(deform.rotations[idx + 1]);
            deformed_rotations.push(deform.rotations[idx + 2]);
            deformed_rotations.push(deform.rotations[idx + 3]);
        } else {
            let base_idx = i * 4;
            deformed_rotations.push(base_rotations[base_idx]);
            deformed_rotations.push(base_rotations[base_idx + 1]);
            deformed_rotations.push(base_rotations[base_idx + 2]);
            deformed_rotations.push(base_rotations[base_idx + 3]);
        }
    }

    let n_deformable_scales = deform.scales.len() / 3;
    let mut deformed_log_scales = Vec::with_capacity(base_log_scales.len());
    for i in 0..total_splats {
        if i < n_deformable_scales {
            let idx = i * 3;
            // TODO: this is hacked together, need to fix later when scale is guaranteed to be positive
            deformed_log_scales.push(deform.scales[idx].abs().max(1e-10).ln());
            deformed_log_scales.push(deform.scales[idx + 1].abs().max(1e-10).ln());
            deformed_log_scales.push(deform.scales[idx + 2].abs().max(1e-10).ln());
        } else {
            let base_idx = i * 3;
            deformed_log_scales.push(base_log_scales[base_idx]);
            deformed_log_scales.push(base_log_scales[base_idx + 1]);
            deformed_log_scales.push(base_log_scales[base_idx + 2]);
        }
    }

    (deformed_means, deformed_rotations, deformed_log_scales)
}

// DFMR (delta) format is not currently supported
// fn apply_delta_deformation(
//     base_means: &[f32],
//     base_rotations: &[f32],
//     base_log_scales: &[f32],
//     deform: &FrameDeformation,
//     total_splats: usize,
//     n_deformable: usize,
// ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
//     let mut deformed_means = Vec::with_capacity(base_means.len());
//     for i in 0..total_splats {
//         let base_idx = i * 3;
//         if i < n_deformable {
//             let delta_idx = i * 3;
//             deformed_means.push(base_means[base_idx] + deform.positions[delta_idx]);
//             deformed_means.push(base_means[base_idx + 1] + deform.positions[delta_idx + 1]);
//             deformed_means.push(base_means[base_idx + 2] + deform.positions[delta_idx + 2]);
//         } else {
//             deformed_means.push(base_means[base_idx]);
//             deformed_means.push(base_means[base_idx + 1]);
//             deformed_means.push(base_means[base_idx + 2]);
//         }
//     }
//
//     let mut deformed_rotations = Vec::with_capacity(base_rotations.len());
//     for i in 0..total_splats {
//         let base_idx = i * 4;
//         if i < n_deformable {
//             let delta_idx = i * 4;
//             let base_q = glam::Quat::from_xyzw(
//                 base_rotations[base_idx + 1],
//                 base_rotations[base_idx + 2],
//                 base_rotations[base_idx + 3],
//                 base_rotations[base_idx], // w is first in our format
//             );
//             let delta_q = glam::Quat::from_xyzw(
//                 deform.rotations[delta_idx + 1],
//                 deform.rotations[delta_idx + 2],
//                 deform.rotations[delta_idx + 3],
//                 deform.rotations[delta_idx], // w is first in our format
//             );
//             let result = (delta_q * base_q).normalize();
//             deformed_rotations.push(result.w);
//             deformed_rotations.push(result.x);
//             deformed_rotations.push(result.y);
//             deformed_rotations.push(result.z);
//         } else {
//             deformed_rotations.push(base_rotations[base_idx]);
//             deformed_rotations.push(base_rotations[base_idx + 1]);
//             deformed_rotations.push(base_rotations[base_idx + 2]);
//             deformed_rotations.push(base_rotations[base_idx + 3]);
//         }
//     }
//
//     (deformed_means, deformed_rotations, base_log_scales.to_vec())
// }

pub fn is_deformation_file(data: &[u8]) -> bool {
    data.len() >= 4 && data.starts_with(MAGIC_DFSS)
}
