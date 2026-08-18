//! Metal backend (macOS / iOS). Renders via the system GPU (Apple silicon,
//! AMD or Intel). Note: Apple's NPU (ANE) is an ML accelerator and cannot
//! render graphics; Metal on the GPU is the acceleration path.

use std::os::raw::c_void;

use core_graphics_types::geometry::CGSize;
pub use metal::MetalLayer;

use metal::{
    CommandQueue, CompileOptions, DepthStencilDescriptor, DepthStencilState, Device,
    MTLBlendFactor, MTLBlendOperation, MTLBlitOption, MTLCompareFunction, MTLClearColor,
    MTLLoadAction, MTLIndexType, MTLPixelFormat, MTLPrimitiveType, MTLOrigin, MTLResourceOptions,
    MTLSize, MTLStoreAction, MTLStorageMode, MTLTextureType, MTLTextureUsage,
    MTLVertexFormat, MTLViewport, MetalDrawableRef, RenderPassDescriptor,
    RenderPipelineDescriptor, RenderPipelineState, TextureDescriptor, VertexDescriptor,
};

use crate::backend::{make_u3d, Draw, Frame, Gpu, U2D};
use crate::error::{Error, Result};
use crate::scene::Camera;

const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct U2D {
    float4 vp;
};

struct V2DOut {
    float4 position [[position]];
    float4 color;
};

vertex V2DOut v2d_vertex(
    float2 pos [[attribute(0)]],
    float4 color [[attribute(1)]],
    constant U2D& u [[buffer(1)]])
{
    V2DOut out;
    // Screen space (y-down) -> NDC (y-up).
    float2 ndc = float2(pos.x / u.vp.x * 2.0 - 1.0, 1.0 - pos.y / u.vp.y * 2.0);
    out.position = float4(ndc, 0.0, 1.0);
    out.color = color;
    return out;
}

fragment float4 v2d_fragment(V2DOut in [[stage_in]]) {
    return in.color;
}

struct U3D {
    float4x4 mvp;
    float4x4 model;
    float3 cam_pos;
    float3 light_dir;
    float4 tint;
    float ambient;
    float specular;
    float _pad;
};

struct V3DOut {
    float4 position [[position]];
    float4 color;
    float3 normal;
    float3 world_pos;
};

vertex V3DOut v3d_vertex(
    float3 pos [[attribute(0)]],
    float3 normal [[attribute(1)]],
    float4 color [[attribute(2)]],
    constant U3D& u [[buffer(1)]])
{
    V3DOut out;
    out.position = u.mvp * float4(pos, 1.0);
    out.world_pos = (u.model * float4(pos, 1.0)).xyz;
    out.normal = normalize((u.model * float4(normal, 0.0)).xyz);
    out.color = color;
    return out;
}

fragment float4 v3d_fragment(V3DOut in [[stage_in]], constant U3D& u [[buffer(0)]]) {
    float3 n = normalize(in.normal);
    float3 l = normalize(u.light_dir);
    float3 v = normalize(u.cam_pos - in.world_pos);
    float diff = max(dot(n, l), 0.0);
    float3 hv = normalize(l + v);
    float spec = pow(max(dot(n, hv), 0.0), 48.0) * u.specular;
    float3 albedo = in.color.rgb * u.tint.rgb;
    float3 col = albedo * (u.ambient + (1.0 - u.ambient) * diff) + spec;
    return float4(col, in.color.a * u.tint.a);
}
"#;

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    pipeline_2d: RenderPipelineState,
    pipeline_3d: RenderPipelineState,
    dss_depth: DepthStencilState,
    dss_none: DepthStencilState,

    width: u32,
    height: u32,

    offscreen_color: Option<metal::Texture>,
    offscreen_depth: Option<metal::Texture>,
    window_layer: Option<MetalLayer>,
}

/// Create a CAMetalLayer and attach it as the backing layer of a winit
/// window's NSView. `ns_view` is the raw `NSView*` (from raw-window-handle's
/// `AppkitWindowHandle::ns_view`).
pub fn attach_layer_to_view(
    ns_view: *mut c_void,
    width: u32,
    height: u32,
) -> Result<MetalLayer> {
    use metal::objc::msg_send;
    use metal::objc::runtime::Object;
    use metal::objc::sel;
    use metal::objc::sel_impl;

    let device =
        Device::system_default().ok_or_else(|| Error::NoGpuDevice("no Metal device".into()))?;
    let layer = MetalLayer::new();
    layer.set_device(&device);
    layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    layer.set_drawable_size(CGSize::new(width as f64, height as f64));
    unsafe {
        let view = ns_view as *mut Object;
        let layer_id: *mut Object = std::mem::transmute(&layer);
        let () = msg_send![view, setWantsLayer: true];
        let () = msg_send![view, setLayer: layer_id];
    }
    Ok(layer)
}

impl MetalBackend {
    fn vertex_descriptor(specs: &[(MTLVertexFormat, u64)]) -> &metal::VertexDescriptorRef {
        let vd = VertexDescriptor::new();
        for (i, (fmt, off)) in specs.iter().enumerate() {
            let a = vd.attributes().object_at(i as u64).unwrap();
            a.set_format(*fmt);
            a.set_offset(*off);
            a.set_buffer_index(0);
        }
        vd
    }

    fn build_pipelines(device: &Device) -> Result<(RenderPipelineState, RenderPipelineState)> {
        metal::objc::rc::autoreleasepool(|| {
            let lib = device
                .new_library_with_source(SHADERS, &CompileOptions::new())
                .map_err(|e| Error::Pipeline(format!("shader compile: {e}")))?;
            let get = |name: &str| {
                lib.get_function(name, None).map_err(|e| Error::Pipeline(format!("{name}: {e}")))
            };

            let vd2d = Self::vertex_descriptor(&[
                (MTLVertexFormat::Float2, 0),
                (MTLVertexFormat::Float4, 8),
            ]);
            let mut d2 = RenderPipelineDescriptor::new();
            let f = get("v2d_vertex")?;
            let g = get("v2d_fragment")?;
            d2.set_vertex_function(Some(&f));
            d2.set_fragment_function(Some(&g));
            d2.set_vertex_descriptor(Some(vd2d));
            {
                let att = d2
                    .color_attachments()
                    .object_at(0)
                    .ok_or_else(|| Error::Pipeline("color attachment".into()))?;
                att.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                att.set_blending_enabled(true);
                att.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
                att.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
                att.set_rgb_blend_operation(MTLBlendOperation::Add);
                att.set_source_alpha_blend_factor(MTLBlendFactor::SourceAlpha);
                att.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
                att.set_alpha_blend_operation(MTLBlendOperation::Add);
            }
            let pipeline_2d = device
                .new_render_pipeline_state(&d2)
                .map_err(|e| Error::Pipeline(format!("2d pipeline: {e}")))?;

            let vd3d = Self::vertex_descriptor(&[
                (MTLVertexFormat::Float3, 0),
                (MTLVertexFormat::Float3, 12),
                (MTLVertexFormat::Float4, 24),
            ]);
            let mut d3 = RenderPipelineDescriptor::new();
            let f = get("v3d_vertex")?;
            let g = get("v3d_fragment")?;
            d3.set_vertex_function(Some(&f));
            d3.set_fragment_function(Some(&g));
            d3.set_vertex_descriptor(Some(vd3d));
            d3.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
            d3.color_attachments()
                .object_at(0)
                .ok_or_else(|| Error::Pipeline("color attachment".into()))?
                .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            let pipeline_3d = device
                .new_render_pipeline_state(&d3)
                .map_err(|e| Error::Pipeline(format!("3d pipeline: {e}")))?;

            Ok((pipeline_2d, pipeline_3d))
        })
    }

    fn make_depth_texture(&self, w: u32, h: u32) -> metal::Texture {
        let mut desc = TextureDescriptor::new();
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_width(w as u64);
        desc.set_height(h as u64);
        desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        desc.set_usage(MTLTextureUsage::RenderTarget);
        desc.set_storage_mode(MTLStorageMode::Private);
        self.device.new_texture(&desc)
    }

    fn make_offscreen_color(&self, w: u32, h: u32) -> metal::Texture {
        let mut desc = TextureDescriptor::new();
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_width(w as u64);
        desc.set_height(h as u64);
        desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        desc.set_usage(MTLTextureUsage::RenderTarget);
        desc.set_storage_mode(MTLStorageMode::Private);
        self.device.new_texture(&desc)
    }

    fn new_uninit(
        device: Device,
        queue: CommandQueue,
        pipeline_2d: RenderPipelineState,
        pipeline_3d: RenderPipelineState,
        dss_depth: DepthStencilState,
        dss_none: DepthStencilState,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            device,
            queue,
            pipeline_2d,
            pipeline_3d,
            dss_depth,
            dss_none,
            width,
            height,
            offscreen_color: None,
            offscreen_depth: None,
            window_layer: None,
        }
    }

    /// Offscreen target (render to a texture, read back for PNG/CI).
    pub fn offscreen(width: u32, height: u32) -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| Error::NoGpuDevice("no Metal device".into()))?;
        let queue = device.new_command_queue();
        let (p2, p3) = Self::build_pipelines(&device)?;
        let (dd, dn) = Self::depth_states(&device)?;
        let mut b = Self::new_uninit(device, queue, p2, p3, dd, dn, width, height);
        b.offscreen_color = Some(b.make_offscreen_color(width, height));
        b.offscreen_depth = Some(b.make_depth_texture(width, height));
        Ok(b)
    }

    /// Windowed target backed by a CAMetalLayer attached to a window view.
    pub fn window(width: u32, height: u32, layer: MetalLayer) -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| Error::NoGpuDevice("no Metal device".into()))?;
        let queue = device.new_command_queue();
        let (p2, p3) = Self::build_pipelines(&device)?;
        let (dd, dn) = Self::depth_states(&device)?;
        let mut b = Self::new_uninit(device, queue, p2, p3, dd, dn, width, height);
        b.window_layer = Some(layer);
        Ok(b)
    }

    fn depth_states(
        device: &Device,
    ) -> Result<(DepthStencilState, DepthStencilState)> {
        let mut dd = DepthStencilDescriptor::new();
        dd.set_depth_compare_function(MTLCompareFunction::Less);
        dd.set_depth_write_enabled(true);
        let dss_depth = device.new_depth_stencil_state(&dd);
        let dss_none = device.new_depth_stencil_state(&DepthStencilDescriptor::new());
        Ok((dss_depth, dss_none))
    }

    fn upload<T: bytemuck::Pod>(&self, data: &[T]) -> metal::Buffer {
        let bytes = bytemuck::cast_slice(data);
        let buf = self
            .device
            .new_buffer(bytes.len().max(4) as u64, MTLResourceOptions::StorageModeShared);
        if !bytes.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buf.contents() as *mut u8,
                    bytes.len(),
                );
            }
        }
        buf
    }
}

impl Gpu for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
        if let Some(layer) = &self.window_layer {
            layer.set_drawable_size(CGSize::new(w as f64, h as f64));
        }
        if self.offscreen_color.is_some() {
            self.offscreen_color = Some(self.make_offscreen_color(w, h));
            self.offscreen_depth = Some(self.make_depth_texture(w, h));
        }
    }

    fn render(&mut self, frame: &Frame, camera: &Camera) -> Result<()> {
        let w = self.width;
        let h = self.height;

        // Acquire this frame's targets.
        let target: FrameTarget = match &self.window_layer {
            Some(layer) => {
                let drawable = layer
                    .next_drawable()
                    .ok_or_else(|| Error::Surface("no drawable".into()))?;
                let depth = self.make_depth_texture(w, h);
                FrameTarget::Window { color: drawable.texture(), depth, drawable }
            }
            None => {
                let color = self
                    .offscreen_color
                    .as_ref()
                    .ok_or_else(|| Error::Backend("no offscreen texture".into()))?;
                let depth = self
                    .offscreen_depth
                    .as_ref()
                    .ok_or_else(|| Error::Backend("no offscreen depth".into()))?;
                FrameTarget::Offscreen { color, depth }
            }
        };

        let mut pass = RenderPassDescriptor::new();
        {
            let ca = pass
                .color_attachments()
                .object_at(0)
                .ok_or_else(|| Error::Backend("color attachment".into()))?;
            ca.set_texture(Some(target.color()));
            ca.set_load_action(MTLLoadAction::Clear);
            ca.set_store_action(MTLStoreAction::Store);
            ca.set_clear_color(MTLClearColor::new(
                frame.clear.r as f64,
                frame.clear.g as f64,
                frame.clear.b as f64,
                frame.clear.a as f64,
            ));
        }
        {
            let da = pass
                .depth_attachment()
                .ok_or_else(|| Error::Backend("depth attachment".into()))?;
            da.set_texture(Some(target.depth()));
            da.set_load_action(MTLLoadAction::Clear);
            da.set_store_action(MTLStoreAction::DontCare);
            da.set_clear_depth(1.0);
        }

        let cmd = self.queue.new_command_buffer();
        {
            let enc = cmd.new_render_command_encoder(&pass);
            enc.set_viewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: w as f64,
                height: h as f64,
                znear: 0.0,
                zfar: 1.0,
            });

            // Keep every uploaded buffer alive until the buffer is committed.
            let mut keepalive: Vec<metal::Buffer> = Vec::new();
            let u2d = U2D { vp: [w as f32, h as f32, 0.0, 0.0] };
            let u2d_buf = self.upload(std::slice::from_ref(&u2d));

            for draw in &frame.draws {
                match draw {
                    Draw::TwoD { verts, indices } => {
                        if verts.is_empty() || indices.is_empty() {
                            continue;
                        }
                        let base = keepalive.len();
                        keepalive.push(self.upload(verts));
                        keepalive.push(self.upload(indices));
                        let vb = &keepalive[base];
                        let ib = &keepalive[base + 1];
                        enc.set_render_pipeline_state(&self.pipeline_2d);
                        enc.set_depth_stencil_state(&self.dss_none);
                        enc.set_vertex_buffer(0, Some(vb), 0);
                        enc.set_vertex_buffer(1, Some(&u2d_buf), 0);
                        enc.draw_indexed_primitives(
                            MTLPrimitiveType::Triangle,
                            indices.len() as u64,
                            MTLIndexType::UInt32,
                            ib,
                            0,
                        );
                    }
                    Draw::ThreeD { mesh, model, material } => {
                        if mesh.is_empty() {
                            continue;
                        }
                        let u = make_u3d(camera, model, material);
                        let base = keepalive.len();
                        keepalive.push(self.upload(&mesh.vertices));
                        keepalive.push(self.upload(&mesh.indices));
                        keepalive.push(self.upload(std::slice::from_ref(&u)));
                        let vb = &keepalive[base];
                        let ib = &keepalive[base + 1];
                        let ub = &keepalive[base + 2];
                        enc.set_render_pipeline_state(&self.pipeline_3d);
                        enc.set_depth_stencil_state(&self.dss_depth);
                        enc.set_vertex_buffer(0, Some(vb), 0);
                        enc.set_vertex_buffer(1, Some(ub), 0);
                        enc.set_fragment_buffer(0, Some(ub), 0);
                        enc.draw_indexed_primitives(
                            MTLPrimitiveType::Triangle,
                            mesh.indices.len() as u64,
                            MTLIndexType::UInt32,
                            ib,
                            0,
                        );
                    }
                }
            }
            enc.end_encoding();
        }

        if let FrameTarget::Window { drawable, .. } = &target {
            cmd.present_drawable(drawable);
        }
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn readback(&mut self) -> Option<Vec<u8>> {
        let color = self.offscreen_color.as_ref()?;
        let w = self.width as usize;
        let h = self.height as usize;
        let byte_count = w * h * 4;
        let buf = self
            .device
            .new_buffer(byte_count as u64, MTLResourceOptions::StorageModeShared);
        let cmd = self.queue.new_command_buffer();
        {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_texture_to_buffer(
                color,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize::new(w as u64, h as u64, 1),
                &buf,
                0,
                (w * 4) as u64,
                byte_count as u64,
                MTLBlitOption::None,
            );
            blit.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
        let data = unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, byte_count) };
        let bgra = data.to_vec();
        // BGRA8 -> RGBA8
        let mut rgba = Vec::with_capacity(bgra.len());
        for px in bgra.chunks_exact(4) {
            rgba.push(px[2]);
            rgba.push(px[1]);
            rgba.push(px[0]);
            rgba.push(px[3]);
        }
        Some(rgba)
    }
}

/// Per-frame render targets (window drawable or offscreen textures).
enum FrameTarget<'a> {
    Window {
        color: &'a metal::TextureRef,
        depth: metal::Texture,
        drawable: &'a MetalDrawableRef,
    },
    Offscreen {
        color: &'a metal::TextureRef,
        depth: &'a metal::TextureRef,
    },
}

impl FrameTarget<'_> {
    fn color(&self) -> &metal::TextureRef {
        match self {
            FrameTarget::Window { color, .. } => color,
            FrameTarget::Offscreen { color, .. } => color,
        }
    }

    fn depth(&self) -> &metal::TextureRef {
        match self {
            FrameTarget::Window { depth, .. } => depth,
            FrameTarget::Offscreen { depth, .. } => depth,
        }
    }
}
