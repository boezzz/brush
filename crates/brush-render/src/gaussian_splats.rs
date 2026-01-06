use crate::{
    SplatForward,
    camera::Camera,
    render_aux::RenderAux,
    sh::{sh_coeffs_for_degree, sh_degree_from_coeffs},
};
use burn::{
    module::{Module, Param, ParamId},
    prelude::Backend,
    tensor::{Tensor, TensorData, TensorPrimitive, activation::sigmoid, s},
};
use glam::Vec3;
use std::sync::Arc;
use tracing::trace_span;

#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub means: Param<Tensor<B, 2>>,
    pub rotations: Param<Tensor<B, 2>>,
    pub log_scales: Param<Tensor<B, 2>>,
    pub sh_coeffs: Param<Tensor<B, 3>>,
    pub raw_opacities: Param<Tensor<B, 1>>,
}

fn norm_vec<B: Backend>(vec: Tensor<B, 2>) -> Tensor<B, 2> {
    let magnitudes =
        Tensor::clamp_min(Tensor::sum_dim(vec.clone().powi_scalar(2), 1).sqrt(), 1e-32);
    vec / magnitudes
}

pub fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

impl<B: Backend> Splats<B> {
    pub fn from_raw(
        pos_data: Vec<f32>,
        rot_data: Vec<f32>,
        scale_data: Vec<f32>,
        coeffs_data: Vec<f32>,
        opac_data: Vec<f32>,
        device: &B::Device,
    ) -> Self {
        let _ = trace_span!("Splats::from_raw").entered();
        let n_splats = pos_data.len() / 3;
        let log_scales = Tensor::from_data(TensorData::new(scale_data, [n_splats, 3]), device);
        let means_tensor = Tensor::from_data(TensorData::new(pos_data, [n_splats, 3]), device);
        let rotations = Tensor::from_data(TensorData::new(rot_data, [n_splats, 4]), device);
        let n_coeffs = coeffs_data.len() / n_splats;
        let sh_coeffs = Tensor::from_data(
            TensorData::new(coeffs_data, [n_splats, n_coeffs / 3, 3]),
            device,
        );
        let raw_opacities =
            Tensor::from_data(TensorData::new(opac_data, [n_splats]), device).require_grad();
        Self::from_tensor_data(
            means_tensor,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
        )
    }

    /// Set the SH degree of this splat to be equal to `sh_degree`
    pub fn with_sh_degree(mut self, sh_degree: u32) -> Self {
        let n_coeffs = sh_coeffs_for_degree(sh_degree) as usize;
        let [n, cur_coeffs, _] = self.sh_coeffs.dims();

        self.sh_coeffs = self.sh_coeffs.map(|coeffs| {
            let device = coeffs.device();
            let tens = if cur_coeffs < n_coeffs {
                Tensor::cat(
                    vec![
                        coeffs,
                        Tensor::zeros([n, n_coeffs - cur_coeffs, 3], &device),
                    ],
                    1,
                )
            } else {
                coeffs.slice(s![.., 0..n_coeffs])
            };
            tens.detach().require_grad()
        });
        self
    }

    pub fn from_tensor_data(
        means: Tensor<B, 2>,
        rotation: Tensor<B, 2>,
        log_scales: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 3>,
        raw_opacity: Tensor<B, 1>,
    ) -> Self {
        assert_eq!(means.dims()[1], 3, "Means must be 3D");
        assert_eq!(rotation.dims()[1], 4, "Rotation must be 4D");
        assert_eq!(log_scales.dims()[1], 3, "Scales must be 3D");

        Self {
            means: Param::initialized(ParamId::new(), means.detach().require_grad()),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs.detach().require_grad()),
            rotations: Param::initialized(ParamId::new(), rotation.detach().require_grad()),
            raw_opacities: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            log_scales: Param::initialized(ParamId::new(), log_scales.detach().require_grad()),
        }
    }

    pub fn opacities(&self) -> Tensor<B, 1> {
        sigmoid(self.raw_opacities.val())
    }

    pub fn scales(&self) -> Tensor<B, 2> {
        self.log_scales.val().exp()
    }

    pub fn num_splats(&self) -> u32 {
        self.means.dims()[0] as u32
    }

    pub fn rotations_normed(&self) -> Tensor<B, 2> {
        norm_vec(self.rotations.val())
    }

    pub fn with_normed_rotations(mut self) -> Self {
        self.rotations = self.rotations.map(|r| norm_vec(r));
        self
    }

    pub fn sh_degree(&self) -> u32 {
        let [_, coeffs, _] = self.sh_coeffs.dims();
        sh_degree_from_coeffs(coeffs as u32)
    }

    pub fn device(&self) -> B::Device {
        self.means.device()
    }

    #[cfg(any(feature = "debug-validation", test))]
    pub fn validate_values(&self) {
        use crate::validation::validate_tensor_val;

        let num_splats = self.num_splats();

        // Validate means (positions)
        validate_tensor_val(&self.means.val(), "means", None, None);

        // Validate raw rotations and normalized rotations
        validate_tensor_val(&self.rotations.val(), "raw_rotations", None, None);
        let rotations = self.rotations_normed();
        validate_tensor_val(&rotations, "normalized_rotations", None, None);

        // Validate pre-activation scales (log_scales) and post-activation scales
        validate_tensor_val(
            &self.log_scales.val(),
            "log_scales",
            Some(-10.0),
            Some(10.0),
        );

        let scales = self.scales();
        validate_tensor_val(&scales, "scales", Some(1e-20), Some(10000.0));

        // Validate SH coefficients
        validate_tensor_val(&self.sh_coeffs.val(), "sh_coeffs", Some(-5.0), Some(5.0));

        // Validate pre-activation opacity (raw_opacity) and post-activation opacity
        validate_tensor_val(
            &self.raw_opacities.val(),
            "raw_opacity",
            Some(-20.0),
            Some(20.0),
        );
        let opacities = self.opacities();
        validate_tensor_val(&opacities, "opacities", Some(0.0), Some(1.0));

        // Range validation if requested
        // Scales should be positive and reasonable
        validate_tensor_val(&scales, "scales", Some(1e-6), Some(100.0));

        // Normalized rotations should have unit magnitude (quaternion)
        let rot_norms = rotations.powi_scalar(2).sum_dim(1).sqrt();
        validate_tensor_val(&rot_norms, "rotation_magnitudes", Some(1e-12), Some(1000.0));

        // Additional logical checks
        assert!(num_splats > 0, "Splats must contain at least one splat");

        let [n_means, dims] = self.means.dims();
        assert_eq!(dims, 3, "Means must be 3D coordinates");
        assert_eq!(
            n_means, num_splats as usize,
            "Inconsistent number of splats in means"
        );
        let [n_rot, rot_dims] = self.rotations.dims();
        assert_eq!(rot_dims, 4, "Rotations must be quaternions (4D)");
        assert_eq!(
            n_rot, num_splats as usize,
            "Inconsistent number of splats in rotations"
        );
        let [n_scales, scale_dims] = self.log_scales.dims();
        assert_eq!(scale_dims, 3, "Scales must be 3D");
        assert_eq!(
            n_scales, num_splats as usize,
            "Inconsistent number of splats in scales"
        );
        let [n_opacity] = self.raw_opacities.dims();
        assert_eq!(
            n_opacity, num_splats as usize,
            "Inconsistent number of splats in opacity"
        );
        let [n_sh, _coeffs, sh_dims] = self.sh_coeffs.dims();
        assert_eq!(sh_dims, 3, "SH coefficients must have 3 color channels");
        assert_eq!(
            n_sh, num_splats as usize,
            "Inconsistent number of splats in SH coeffs"
        );
    }
}

impl<B: Backend + SplatForward<B>> Splats<B> {
    /// Render the splats.
    ///
    /// NB: This doesn't work on a differentiable backend.
    pub fn render(
        &self,
        camera: &Camera,
        img_size: glam::UVec2,
        background: Vec3,
        splat_scale: Option<f32>,
    ) -> (Tensor<B, 3>, RenderAux<B>) {
        let mut scales = self.log_scales.val();

        #[cfg(any(feature = "debug-validation", test))]
        self.validate_values();

        // Add in scaling if needed.
        if let Some(scale) = splat_scale {
            scales = scales + scale.ln();
        };

        let (img, aux) = B::render_splats(
            camera,
            img_size,
            self.means.val().into_primitive().tensor(),
            scales.into_primitive().tensor(),
            self.rotations.val().into_primitive().tensor(),
            self.sh_coeffs.val().into_primitive().tensor(),
            self.raw_opacities.val().into_primitive().tensor(),
            background,
            false,
        );
        let img = Tensor::from_primitive(TensorPrimitive::Float(img));
        #[cfg(any(feature = "debug-validation", test))]
        aux.validate_values();
        (img, aux)
    }
}

/// animated splats that share static data across frames
#[derive(Debug, Clone)]
pub struct AnimatedSplats<B: Backend> {
    pub means: Tensor<B, 3>,
    pub rotations: Tensor<B, 3>,
    pub log_scales: Tensor<B, 3>,
    pub sh_coeffs: Arc<Tensor<B, 3>>,
    pub raw_opacities: Arc<Tensor<B, 1>>,
    pub num_frames: u32,
}

impl<B: Backend> AnimatedSplats<B> {
    pub fn from_separated(
        dynamic_means: Vec<f32>,
        dynamic_rotations: Vec<f32>,
        dynamic_log_scales: Vec<f32>,
        static_means: Vec<f32>,
        static_rotations: Vec<f32>,
        static_log_scales: Vec<f32>,
        sh_coeffs: Vec<f32>,
        raw_opacities: Vec<f32>,
        num_frames: u32,
        num_deformable: usize,
        num_static: usize,
        device: &B::Device,
    ) -> Self {
        let _ = trace_span!("AnimatedSplats::from_separated").entered();

        let n_frames = num_frames as usize;
        let num_splats = num_deformable + num_static;

        // build tensors directly on GPU to avoid large CPU allocations
        let dynamic_means_tensor = Tensor::from_data(
            TensorData::new(dynamic_means, [n_frames, num_deformable, 3]),
            device,
        );
        let dynamic_rotations_tensor = Tensor::from_data(
            TensorData::new(dynamic_rotations, [n_frames, num_deformable, 4]),
            device,
        );
        let dynamic_log_scales_tensor = Tensor::from_data(
            TensorData::new(dynamic_log_scales, [n_frames, num_deformable, 3]),
            device,
        );

        let means = if num_static > 0 {
            let static_means_tensor: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(static_means, [num_static, 3]),
                device,
            );
            let static_means_tensor = static_means_tensor.unsqueeze_dim(0).repeat_dim(0, n_frames);
            Tensor::cat(vec![dynamic_means_tensor, static_means_tensor], 1)
        } else {
            dynamic_means_tensor
        };

        let rotations = if num_static > 0 {
            let static_rotations_tensor: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(static_rotations, [num_static, 4]),
                device,
            );
            let static_rotations_tensor = static_rotations_tensor.unsqueeze_dim(0).repeat_dim(0, n_frames);
            Tensor::cat(vec![dynamic_rotations_tensor, static_rotations_tensor], 1)
        } else {
            dynamic_rotations_tensor
        };

        let log_scales = if num_static > 0 {
            let static_log_scales_tensor: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(static_log_scales, [num_static, 3]),
                device,
            );
            let static_log_scales_tensor = static_log_scales_tensor.unsqueeze_dim(0).repeat_dim(0, n_frames);
            Tensor::cat(vec![dynamic_log_scales_tensor, static_log_scales_tensor], 1)
        } else {
            dynamic_log_scales_tensor
        };

        let n_coeffs = sh_coeffs.len() / num_splats;
        let sh_coeffs = Arc::new(Tensor::from_data(
            TensorData::new(sh_coeffs, [num_splats, n_coeffs / 3, 3]),
            device,
        ));
        let raw_opacities = Arc::new(Tensor::from_data(
            TensorData::new(raw_opacities, [num_splats]),
            device,
        ));

        Self {
            means,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
            num_frames,
        }
    }

    pub fn from_raw(
        all_means: Vec<f32>,
        all_rotations: Vec<f32>,
        all_log_scales: Vec<f32>,
        sh_coeffs: Vec<f32>,
        raw_opacities: Vec<f32>,
        num_frames: u32,
        num_splats: usize,
        device: &B::Device,
    ) -> Self {
        let _ = trace_span!("AnimatedSplats::from_raw").entered();

        let n_frames = num_frames as usize;

        let means = Tensor::from_data(
            TensorData::new(all_means, [n_frames, num_splats, 3]),
            device,
        );
        let rotations = Tensor::from_data(
            TensorData::new(all_rotations, [n_frames, num_splats, 4]),
            device,
        );
        let log_scales = Tensor::from_data(
            TensorData::new(all_log_scales, [n_frames, num_splats, 3]),
            device,
        );

        let n_coeffs = sh_coeffs.len() / num_splats;
        let sh_coeffs = Arc::new(Tensor::from_data(
            TensorData::new(sh_coeffs, [num_splats, n_coeffs / 3, 3]),
            device,
        ));
        let raw_opacities = Arc::new(Tensor::from_data(
            TensorData::new(raw_opacities, [num_splats]),
            device,
        ));

        Self {
            means,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
            num_frames,
        }
    }

    pub fn num_splats(&self) -> u32 {
        self.means.dims()[1] as u32
    }

    pub fn sh_degree(&self) -> u32 {
        let [_, coeffs, _] = self.sh_coeffs.dims();
        sh_degree_from_coeffs(coeffs as u32)
    }

    pub fn get_frame(&self, frame: u32) -> Splats<B> {
        let frame = if self.num_frames > 0 {
            frame.min(self.num_frames - 1) as usize
        } else {
            0
        };

        let num_splats = self.means.dims()[1];

        let means = self
            .means
            .clone()
            .slice(s![frame..frame + 1, .., ..])
            .reshape([num_splats, 3]);
        let rotations = self
            .rotations
            .clone()
            .slice(s![frame..frame + 1, .., ..])
            .reshape([num_splats, 4]);
        let log_scales = self
            .log_scales
            .clone()
            .slice(s![frame..frame + 1, .., ..])
            .reshape([num_splats, 3]);

        let sh_coeffs = (*self.sh_coeffs).clone();
        let raw_opacities = (*self.raw_opacities).clone();

        Splats::from_tensor_data(means, rotations, log_scales, sh_coeffs, raw_opacities)
    }

    pub fn device(&self) -> B::Device {
        self.means.device()
    }
}
