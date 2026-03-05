vulkano_shaders::shader! {
    ty: "compute",
    path: "src/shader/sha256.comp",
}

pub fn load_pipeline() {
    // Disabled compilation of pipeline due to vulkano versions mapping.
    // Vulkan dispatch relies on mock dummy loop.
}
