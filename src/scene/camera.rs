use crate::math::{Mat4, Vec3};

/// Right-handed perspective camera. Metal/Vulkan clip depth is [0, 1].
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_deg: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_deg: 60.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    #[must_use]
    pub fn look_at(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            eye,
            target,
            up,
            fov_y_deg,
            aspect,
            near,
            far,
        }
    }

    /// View matrix (world -> camera space).
    #[must_use]
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    /// Perspective projection matrix.
    #[must_use]
    pub fn projection(&self) -> Mat4 {
        let fov = self.fov_y_deg.to_radians().max(1e-4);
        Mat4::perspective_rh(fov, self.aspect.max(1e-4), self.near, self.far)
    }

    /// `projection * view`.
    #[must_use]
    pub fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect.max(1e-4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec4;

    #[test]
    fn target_projects_to_clip_origin() {
        let cam = Camera::default();
        let vp = cam.view_projection();
        let clip = vp * Vec4::from((0.0f32, 0.0, 0.0, 1.0));
        let ndc = clip / clip.w;
        assert!(ndc.x.abs() < 1e-4);
        assert!(ndc.y.abs() < 1e-4);
        // Eye at z=3 looking at origin: target is between near (0.1) and far (100),
        // so NDC z is in (0, 1) for the Metal [0, 1] depth range.
        assert!((0.0..1.0).contains(&ndc.z));
    }
}
