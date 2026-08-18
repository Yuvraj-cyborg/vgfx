use bytemuck::{Pod, Zeroable};

/// GPU vertex for 3D rendering: position, normal, RGBA color.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex3D {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// Indexed triangle mesh (right-handed, CCW front faces).
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

impl Mesh {
    /// Build a mesh, computing the axis-aligned bounds.
    #[must_use]
    pub fn new(vertices: Vec<Vertex3D>, indices: Vec<u32>) -> Self {
        let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &vertices {
            for i in 0..3 {
                min[i] = min[i].min(v.pos[i]);
                max[i] = max[i].max(v.pos[i]);
            }
        }
        if vertices.is_empty() {
            min = [0.0; 3];
            max = [0.0; 3];
        }
        Self { vertices, indices, bounds_min: min, bounds_max: max }
    }

    /// Build from flat arrays. `positions`/`normals` in xyz triples,
    /// `colors` in rgba quads. Returns None on length mismatch.
    #[must_use]
    pub fn from_data(
        positions: &[f32],
        normals: &[f32],
        colors: &[f32],
        indices: &[u32],
    ) -> Option<Self> {
        let count = positions.len() / 3;
        if positions.len() % 3 != 0
            || normals.len() != positions.len()
            || colors.len() != count * 4
            || indices.iter().any(|&i| i as usize >= count)
        {
            return None;
        }
        let vertices = (0..count)
            .map(|i| Vertex3D {
                pos: [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]],
                normal: [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]],
                color: [colors[i * 4], colors[i * 4 + 1], colors[i * 4 + 2], colors[i * 4 + 3]],
            })
            .collect();
        Some(Self::new(vertices, indices.to_vec()))
    }

    /// Unit cube scaled to `size`, one vertex set per face.
    #[must_use]
    pub fn cube(size: f32) -> Self {
        let h = size * 0.5;
        // (center, normal, u, v, color) per face.
        let faces: [([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 4]); 6] = [
            ([0.0, 0.0, h], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.85, 0.35, 0.35, 1.0]), // +z
            ([0.0, 0.0, -h], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.75, 0.30, 0.35, 1.0]), // -z
            ([h, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.35, 0.75, 0.85, 1.0]), // +x
            ([-h, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.30, 0.65, 0.75, 1.0]), // -x
            ([0.0, h, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.75, 0.85, 0.35, 1.0]), // +y
            ([0.0, -h, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.35, 0.75, 0.35, 1.0]), // -y
        ];
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        for (center, n, u, v, color) in faces {
            let base = vertices.len() as u32;
            for (su, sv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                vertices.push(Vertex3D {
                    pos: [
                        center[0] + (u[0] * su + v[0] * sv) * h,
                        center[1] + (u[1] * su + v[1] * sv) * h,
                        center[2] + (u[2] * su + v[2] * sv) * h,
                    ],
                    normal: n,
                    color: color,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self::new(vertices, indices)
    }

    /// UV sphere (width x height segments).
    #[must_use]
    pub fn sphere(radius: f32, width_segments: u32, height_segments: u32) -> Self {
        let w = width_segments.max(3) as u32;
        let h = height_segments.max(2) as u32;
        let mut vertices = Vec::with_capacity((w + 1) as usize * (h + 1) as usize);
        for iy in 0..=h {
            let theta = iy as f32 / h as f32 * std::f32::consts::PI;
            let (sin_t, cos_t) = theta.sin_cos();
            for ix in 0..=w {
                let phi = ix as f32 / w as f32 * std::f32::consts::TAU;
                let (sin_p, cos_p) = phi.sin_cos();
                let n = glam::Vec3::new(sin_t * cos_p, cos_t, sin_t * sin_p);
                let p = n * radius;
                vertices.push(Vertex3D {
                    pos: [p.x, p.y, p.z],
                    normal: [n.x, n.y, n.z],
                    color: [1.0, 1.0, 1.0, 1.0],
                });
            }
        }
        let mut indices = Vec::with_capacity(w as usize * h as usize * 6);
        for iy in 0..h {
            for ix in 0..w {
                let a = (iy * (w + 1) + ix) as u32;
                let b = a + w + 1;
                indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        Self::new(vertices, indices)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    #[must_use]
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    #[must_use]
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.bounds_min, self.bounds_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_structure() {
        let m = Mesh::cube(2.0);
        assert_eq!(m.vertex_count(), 24);
        assert_eq!(m.index_count(), 36);
        let (min, max) = m.bounds();
        assert!((min[0] - (-1.0)).abs() < 1e-6);
        assert!((max[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_on_surface() {
        let m = Mesh::sphere(3.0, 16, 12);
        assert_eq!(m.vertex_count(), 17 * 13);
        assert_eq!(m.index_count(), 16 * 12 * 6);
        for v in &m.vertices {
            let d = (v.pos[0] * v.pos[0] + v.pos[1] * v.pos[1] + v.pos[2] * v.pos[2]).sqrt();
            assert!((d - 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn from_data_validates() {
        let positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let normals = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let colors = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let indices = [0u32, 1, 2];
        let m = Mesh::from_data(&positions, &normals, &colors, &indices).unwrap();
        assert_eq!(m.vertex_count(), 3);
        assert!(Mesh::from_data(&positions, &normals, &colors, &[9]).is_none());
        assert!(Mesh::from_data(&positions, &normals, &[], &indices).is_none());
    }
}
