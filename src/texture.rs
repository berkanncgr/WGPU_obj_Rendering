use anyhow::*;
use image::GenericImageView;
use std::{num::NonZeroU32, path::Path};

pub struct Texture
{
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture
{
    // We need the DEPTH_FORMAT for when we create the depth stage of the render_pipeline and creating the depth texture itself.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn load<P: AsRef<Path>>(device: &wgpu::Device,queue: &wgpu::Queue,path: P,)-> Result<Self>
    {
        // Needed to appease the borrow checker
        let path_copy = path.as_ref().to_path_buf();
        let label = path_copy.to_str();
        let img = image::open(path)?;
        println!(" image loaded success: texture.rs");

        //Return value.
        Self::from_image(device, queue, &img, label)
    }

    // Create texture.
    pub fn create_depth_texture(device: &wgpu::Device,sc_desc: &wgpu::SwapChainDescriptor,label: &str,)-> Self
    {   
        let size = wgpu::Extent3d
        {
            width: sc_desc.width,
            height: sc_desc.height,
            depth_or_array_layers: 1,
        };

        let desc = wgpu::TextureDescriptor
        {
            // All textures are stored as 3D, we represent our 2D texture by setting depth to 1.
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Most images are stored using sRGB so we need to reflect that here.
            format: Self::DEPTH_FORMAT,
            // SAMPLED tells wgpu that we want to use this texture in shaders.
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        };

        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // A TextureView offers us a view into our texture. A Sampler controls how the Texture is sampled.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor
        {
            // We don't need to configure the texture view much, so let's let wgpu define it.
            // Adrdress_modes etermines what to do if the sampler gets a texture coordinate that's outside of the texture itself.
            // ClampToEdge: Any texture coordinates outside the texture will return the color of the nearest pixel on the edges of the texture.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,

            // The mag_filter and min_filter options describe what to do when a fragment covers multiple pixels,
            // or there are multiple fragments for a single pixel.
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        // Return Value.
        Self { texture, view, sampler, }
    }

    #[allow(dead_code)]
    pub fn from_bytes(device: &wgpu::Device,queue: &wgpu::Queue,bytes: &[u8],label: &str,)-> Result<Self>
    {
        let img = image::load_from_memory(bytes)?;

        // Return value.
        Self::from_image(device, queue, &img, Some(label))
    }

    pub fn from_image(device: &wgpu::Device,queue: &wgpu::Queue,img: &image::DynamicImage,label: Option<&str>,)-> Result<Self>
    {
        let dimensions = img.dimensions();
        let rgba = img.to_rgba8();

        let size = wgpu::Extent3d
        {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor
        {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        // we can use a method on the queue we created earlier called write_texture to load the texture.
        queue.write_texture( wgpu::ImageCopyTexture
            {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,   // The actual pixel data

            wgpu::ImageDataLayout
            {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor
        {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Return value.
        Ok(Self { texture, view, sampler, })
    }
}