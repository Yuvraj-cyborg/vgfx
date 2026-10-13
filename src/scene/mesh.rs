/// GPU vertex for 3D rendering: position, normal, RGBA color.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
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
        Self {
            vertices,
            indices,
            bounds_min: min,
            bounds_max: max,
        }
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
        if !positions.len().is_multiple_of(3)
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
                color: [
                    colors[i * 4],
                    colors[i * 4 + 1],
                    colors[i * 4 + 2],
                    colors[i * 4 + 3],
                ],
            })
            .collect();
        Some(Self::new(vertices, indices.to_vec()))
    }

    /// Unit cube scaled to `size`, one vertex set per face.
    #[must_use]
    pub fn cube(size: f32) -> Self {
        let h = size * 0.5;
        /// One cube face: center, normal, tangent axes, base color.
        type Face = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 4]);
        let faces: [Face; 6] = [
            (
                [0.0, 0.0, h],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.85, 0.35, 0.35, 1.0],
            ), // +z
            (
                [0.0, 0.0, -h],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.75, 0.30, 0.35, 1.0],
            ), // -z
            (
                [h, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [0.35, 0.75, 0.85, 1.0],
            ), // +x
            (
                [-h, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.30, 0.65, 0.75, 1.0],
            ), // -x
            (
                [0.0, h, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.75, 0.85, 0.35, 1.0],
            ), // +y
            (
                [0.0, -h, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.35, 0.75, 0.35, 1.0],
            ), // -y
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
                    color,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self::new(vertices, indices)
    }

    /// UV sphere (width x height segments).
    #[must_use]
    pub fn sphere(radius: f32, width_segments: u32, height_segments: u32) -> Self {
        let w = width_segments.max(3);
        let h = height_segments.max(2);
        let mut vertices = Vec::with_capacity((w + 1) as usize * (h + 1) as usize);
        for iy in 0..=h {
            let theta = iy as f32 / h as f32 * std::f32::consts::PI;
            let (sin_t, cos_t) = theta.sin_cos();
            for ix in 0..=w {
                let phi = ix as f32 / w as f32 * std::f32::consts::TAU;
                let (sin_p, cos_p) = phi.sin_cos();
                let n = crate::math::Vec3::new(sin_t * cos_p, cos_t, sin_t * sin_p);
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
                let a = iy * (w + 1) + ix;
                let b = a + w + 1;
                indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        Self::new(vertices, indices)
    }

    /// Load a Wavefront OBJ file (positions + faces; normals computed when
    /// missing, all vertices white so you can tint via [`Material`]).
    ///
    /// All models in the file are merged into one mesh. Only `v`/`f` data
    /// is used — materials, textures, and groups are ignored.
    pub fn load_obj(path: impl AsRef<std::path::Path>) -> crate::error::Result<Self> {
        Self::parse_obj(&std::fs::read_to_string(path)?)
    }

    /// Parse positions, normals and convex polygon faces. Import is an explicit
    /// preparation operation; no hashing, material loading or frame-time parsing.
    pub fn parse_obj(source: &str) -> crate::Result<Self> {
        use crate::Error;
        use crate::math::Vec3;
        let invalid = || Error::Io("invalid OBJ geometry/index".into());
        let mut positions = Vec::with_capacity(
            source
                .lines()
                .filter(|l| l.trim_start().starts_with("v "))
                .count(),
        );
        let mut normals = Vec::new();
        for line in source.lines() {
            let mut words = line.split('#').next().unwrap_or("").split_whitespace();
            let kind = words.next();
            if kind == Some("v") || kind == Some("vn") {
                let mut v = [0.0; 3];
                for x in &mut v {
                    *x = words
                        .next()
                        .ok_or_else(invalid)?
                        .parse::<f32>()
                        .map_err(|_| invalid())?;
                }
                if !v.iter().all(|x| x.is_finite()) {
                    return Err(invalid());
                }
                if kind == Some("v") {
                    positions.push(v);
                } else {
                    normals.push(v);
                }
            }
        }
        let mut vertices: Vec<Vertex3D> = positions
            .iter()
            .map(|&pos| Vertex3D {
                pos,
                normal: [0.0; 3],
                color: [1.0; 4],
            })
            .collect();
        let face_count = source
            .lines()
            .filter(|l| l.trim_start().starts_with("f "))
            .count();
        let mut indices = Vec::with_capacity(face_count.saturating_mul(3));
        let mut face = Vec::new();
        let mut seen_positions = 0;
        let mut seen_normals = 0;
        fn index(text: &str, total: usize, seen: usize) -> Option<usize> {
            let i = text.parse::<isize>().ok()?;
            let i = if i > 0 {
                i - 1
            } else if i < 0 {
                seen as isize + i
            } else {
                return None;
            };
            (i >= 0 && (i as usize) < total).then_some(i as usize)
        }
        for line in source.lines() {
            let mut words = line.split('#').next().unwrap_or("").split_whitespace();
            match words.next() {
                Some("v") => seen_positions += 1,
                Some("vn") => seen_normals += 1,
                Some("f") => {
                    face.clear();
                    for word in words {
                        let mut parts = word.split('/');
                        let p = index(parts.next().unwrap_or(""), positions.len(), seen_positions)
                            .ok_or_else(invalid)?;
                        let _uv = parts.next();
                        let n = parts.next().filter(|s| !s.is_empty());
                        if parts.next().is_some() {
                            return Err(invalid());
                        }
                        let v = if let Some(n) = n {
                            let n = index(n, normals.len(), seen_normals).ok_or_else(invalid)?;
                            let i = u32::try_from(vertices.len()).map_err(|_| invalid())?;
                            let normal = Vec3::from(normals[n]);
                            if normal.length_squared() <= 1e-12 {
                                return Err(invalid());
                            }
                            vertices.push(Vertex3D {
                                pos: positions[p],
                                normal: normal.normalize().to_array(),
                                color: [1.0; 4],
                            });
                            i
                        } else {
                            u32::try_from(p).map_err(|_| invalid())?
                        };
                        face.push(v);
                    }
                    if face.len() < 3 {
                        return Err(invalid());
                    }
                    // Fan triangulation requires a planar convex polygon. Reject
                    // concave/nonplanar n-gons rather than silently covering gaps.
                    if face.len() > 3 {
                        let at = |i: usize| Vec3::from(vertices[face[i % face.len()] as usize].pos);
                        let mut normal = Vec3::ZERO;
                        for i in 0..face.len() {
                            normal += at(i).cross(at(i + 1));
                        }
                        if normal.length_squared() <= 1e-12 {
                            return Err(invalid());
                        }
                        normal = normal.normalize();
                        for i in 0..face.len() {
                            if (at(i + 1) - at(i)).cross(at(i + 2) - at(i + 1)).dot(normal) < -1e-6
                                || (at(i) - at(0)).dot(normal).abs() > 1e-4
                            {
                                return Err(Error::Io("OBJ n-gon must be planar and convex; triangulate it before import".into()));
                            }
                        }
                    }
                    for i in 1..face.len() - 1 {
                        indices.extend_from_slice(&[face[0], face[i], face[i + 1]]);
                    }
                }
                _ => {}
            }
        }
        if indices.is_empty() {
            return Err(Error::Io("obj file has no faces".into()));
        }
        // Missing normals use area-weighted smoothing across position indices.
        for tri in indices.chunks_exact(3) {
            let a = Vec3::from(vertices[tri[0] as usize].pos);
            let n = (Vec3::from(vertices[tri[1] as usize].pos) - a)
                .cross(Vec3::from(vertices[tri[2] as usize].pos) - a);
            for &i in tri {
                if (i as usize) < positions.len() {
                    let v = &mut vertices[i as usize];
                    v.normal = (Vec3::from(v.normal) + n).to_array();
                }
            }
        }
        for v in &mut vertices[..positions.len()] {
            let n = Vec3::from(v.normal);
            v.normal = if n.length_squared() > 1e-12 {
                n.normalize().to_array()
            } else {
                Vec3::Y.to_array()
            };
        }
        Ok(Self::new(vertices, indices))
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

#[cfg(test)]
mod obj_tests {
    use super::*;
    #[test]
    fn obj_negative_indices_and_smoothing() {
        let m = Mesh::parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n").unwrap();
        assert_eq!(m.indices, [0, 1, 2, 0, 2, 3]);
        assert!(m.vertices.iter().all(|v| v.normal == [0., 0., 1.]));
    }
    #[test]
    fn obj_corner_normals_and_bad_indices() {
        let m =
            Mesh::parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 -2\nf 1//1 2//1 3//1\n").unwrap();
        assert!(
            m.indices
                .iter()
                .all(|&i| m.vertices[i as usize].normal == [0., 0., -1.])
        );
        for source in [
            "v 0 0 0\nf 0 1 1",
            "v 0 0 0\nf 1 2 3",
            "v NaN 0 0",
            "v 0 0 0\nf -2 -1 -1",
        ] {
            assert!(Mesh::parse_obj(source).is_err());
        }
    }
    #[test]
    fn bundled_asset_loads() {
        let mesh = Mesh::load_obj(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/ogre.obj")).unwrap();
        assert!(!mesh.is_empty());
        assert!(
            mesh.indices
                .iter()
                .all(|&i| (i as usize) < mesh.vertices.len())
        );
    }
}
