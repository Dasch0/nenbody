use crate::renderer::*;

pub enum Id {
    Triangle = 0,
}

pub const VERTEX_DATA: [&[Vertex]; 1] = [
    &[  // Triangle
        vert([-1, -1, 0], [0, 0]),
        vert([1, 0, 0], [0, 1]),
        vert([1, -1, 0], [0, 0]),
    ],
];

pub const INDEX_DATA: [&[u16]; 1] = [
    &[0, 1, 2, 0],
];

pub const TEXTURE_PATH: &[&str] = &[
    "../assets/triangle_skin.bmp"
];
