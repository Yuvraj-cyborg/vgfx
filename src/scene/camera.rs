use glam::{Mat4, Vec3};

/// Right-handed perspective camera (OpenGL/Metal/Vulkan convention:
/// +z points out of the screen, clip z in [-1, 1]).
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
        Self { eye, target, up, fov_y_deg, aspect, near, far }
    }

    /// View matrix (world -> camera space).
    #[must_use]
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    /// Perspective projection matrix (clip z in [-1, 1]).
    #[must_use]
    pub fn projection(&self) -> Mat4 {
        let fov = self.fov_y_deg.to_radians().max(1e-4);
        Mat4::perspective_rh_gl(fov, self.aspect.max(1e-4), self.near, self.far)
    }

    /// `projection * view`.
    #[must_use]
    pub fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }

    /// Normal matrix for `view * model`: transposed inverse of the
    /// upper-left 3x3 (glam Mat3, column-major).
    #[must_use]
    pub fn normal_matrix(&self, model: &Mat4) -> glam::Mat3 {
        let m = self.view() * model;
        let nm = glam::Mat3 { x_axis: m.x_axis.truncate(), y_axis: m.y_axis.truncate(), z_axis: m.z_axis.truncate() };
        nm.inverse().transpose()
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect.max(1e-4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn target_projects_to_clip_origin() {
        let cam = Camera::default();
        let vp = cam.view_projection();
        let clip = vp * Vec4::from((0.0f32, 0.0, 0.0, 1.0));
        let ndc = clip / clip.w;
        assert!(ndc.x.abs() < 1e-4);
        assert!(ndc.y.abs() < 1e-4);
        // Eye at z=3 looking at origin: target is between near (0.1) and far (100),
        // so NDC z is in (0, 1) for the GL [-1, 1] depth range.
        assert!((0.0..1.0).contains(&ndc.z));
    }

    #[test]
    fn normal_matrix_is_orthonormal_for_rigid() {
        let cam = Camera::default();
        let model = Mat4::from_rotation_y(0.7);
        let nm = cam.normal_matrix(&model);
        // Column 0 should be unit length.
        assert!((nm.x_axis.length() - 1.0).abs() < 1e-4);
    }
}
