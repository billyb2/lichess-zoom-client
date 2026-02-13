use image::codecs::jpeg::JpegEncoder;
use image::imageops::{FilterType, overlay, resize};
use image::{ColorType, DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use rayon::prelude::*;
use shakmaty::{Board, Chess, Color, File, Position, Rank, Role, Square};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const BOARD_DIMENSION: u32 = 8;
const TILE_DIMENSION_SQUARES: u32 = 2;
const TILE_GRID_DIMENSION: u32 = BOARD_DIMENSION / TILE_DIMENSION_SQUARES;
const DEFAULT_SQUARE_SIZE: u32 = 128;
const DEFAULT_JPEG_QUALITY: u8 = 90;
const OUTPUT_FRAME_WIDTH: u32 = 1280;
const OUTPUT_FRAME_HEIGHT: u32 = 720;
const RAW_TILE_SQUARE_SIZE: u32 = OUTPUT_FRAME_HEIGHT / TILE_DIMENSION_SQUARES;
const RAW_TILE_SIDE: u32 = RAW_TILE_SQUARE_SIZE * TILE_DIMENSION_SQUARES;
const OUTPUT_TILE_SQUARE_SIZE: u32 = OUTPUT_FRAME_HEIGHT / TILE_DIMENSION_SQUARES;
const LIGHT_SQUARE: [u8; 3] = [240, 217, 181];
const DARK_SQUARE: [u8; 3] = [181, 136, 99];
const TILE_COUNT: usize = (TILE_GRID_DIMENSION * TILE_GRID_DIMENSION) as usize;

#[derive(Debug)]
struct RenderTimings {
    tile_render_ms: u128,
    wall_compose_ms: u128,
    jpeg_encode_ms: u128,
    total_ms: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PieceSpriteCacheKey {
    piece_index: u8,
    square_size_px: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileBaseCacheKey {
    tile_index: usize,
    square_size_px: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WallLayoutKey {
    columns: u32,
    rows: u32,
}

static PIECE_SPRITE_CACHE: OnceLock<Mutex<HashMap<PieceSpriteCacheKey, Arc<RgbaImage>>>> =
    OnceLock::new();
static TILE_BASE_CACHE: OnceLock<Mutex<HashMap<TileBaseCacheKey, Arc<RgbaImage>>>> =
    OnceLock::new();
static WALL_BACKGROUND_CACHE: OnceLock<Mutex<HashMap<WallLayoutKey, Arc<RgbaImage>>>> =
    OnceLock::new();
static OUTPUT_FRAME_BACKGROUND_CACHE: OnceLock<Arc<RgbaImage>> = OnceLock::new();
static NEXT_WALL_RENDER_FRAME_ID: AtomicU64 = AtomicU64::new(1);

pub fn render_default_empty_board_jpeg() -> Result<Vec<u8>, image::ImageError> {
    render_empty_board_jpeg(DEFAULT_SQUARE_SIZE, DEFAULT_JPEG_QUALITY)
}

pub fn render_empty_board_jpeg(
    square_size_px: u32,
    jpeg_quality: u8,
) -> Result<Vec<u8>, image::ImageError> {
    let board_size = BOARD_DIMENSION * square_size_px;
    let mut image = RgbImage::new(board_size, board_size);

    for rank_from_top in 0..BOARD_DIMENSION {
        for file in 0..BOARD_DIMENSION {
            let board_rank = board_rank_from_top(rank_from_top);
            let square_color = square_color(file, board_rank);
            paint_square_rgb(
                &mut image,
                file,
                rank_from_top,
                square_size_px,
                Rgb(square_color),
            );
        }
    }

    encode_rgb_to_jpeg(&image, jpeg_quality)
}

pub fn render_default_board_tile_jpeg(tile_index: usize) -> Result<Vec<u8>, image::ImageError> {
    let position = Chess::default();
    render_board_tile_jpeg(
        position.board(),
        tile_index,
        DEFAULT_SQUARE_SIZE,
        DEFAULT_JPEG_QUALITY,
        None,
        None,
        None,
        None,
    )
}

pub fn render_current_board_tile_jpeg(
    board: &Board,
    tile_index: usize,
    white_player_name: Option<&str>,
    black_player_name: Option<&str>,
    white_chat_message: Option<&str>,
    black_chat_message: Option<&str>,
) -> Result<Vec<u8>, image::ImageError> {
    render_board_tile_jpeg(
        board,
        tile_index,
        OUTPUT_TILE_SQUARE_SIZE,
        DEFAULT_JPEG_QUALITY,
        white_player_name,
        black_player_name,
        white_chat_message,
        black_chat_message,
    )
}

pub fn render_current_board_wall_tile_jpegs(
    board: &Board,
    output_tile_count: usize,
    white_player_name: Option<&str>,
    black_player_name: Option<&str>,
    white_chat_message: Option<&str>,
    black_chat_message: Option<&str>,
) -> Result<Vec<Vec<u8>>, image::ImageError> {
    if output_tile_count == 0 {
        return Ok(Vec::new());
    }
    let (wall_columns, wall_rows) = wall_layout(output_tile_count);
    let frame_id = NEXT_WALL_RENDER_FRAME_ID.fetch_add(1, Ordering::Relaxed);
    let wall_started_at = Instant::now();

    let (rendered_tiles, tile_render_ms) = render_board_tiles_for_wall(
        board,
        white_player_name,
        black_player_name,
        white_chat_message,
        black_chat_message,
    )?;

    let wall_compose_started_at = Instant::now();
    let wall_frame = compose_wall_frame_cpu(rendered_tiles, wall_columns, wall_rows);
    let wall_compose_ms = wall_compose_started_at.elapsed().as_millis();

    let jpeg_encode_started_at = Instant::now();
    let tile_jpegs =
        encode_wall_tiles_to_jpeg_cpu(&wall_frame, wall_columns, output_tile_count)?;
    let jpeg_encode_ms = jpeg_encode_started_at.elapsed().as_millis();

    let timings = RenderTimings {
        tile_render_ms,
        wall_compose_ms,
        jpeg_encode_ms,
        total_ms: wall_started_at.elapsed().as_millis(),
    };
    log_render_timings(frame_id, &timings);

    Ok(tile_jpegs)
}

fn wall_layout(output_tile_count: usize) -> (u32, u32) {
    let columns = (output_tile_count as f64).sqrt().ceil() as u32;
    let rows = ((output_tile_count as u32) + columns - 1) / columns;
    (columns.max(1), rows.max(1))
}

fn wall_frame_dimensions(wall_columns: u32, wall_rows: u32) -> (u32, u32) {
    (
        OUTPUT_FRAME_WIDTH * wall_columns.max(1),
        OUTPUT_FRAME_HEIGHT * wall_rows.max(1),
    )
}

fn render_board_tiles_for_wall(
    board: &Board,
    white_player_name: Option<&str>,
    black_player_name: Option<&str>,
    white_chat_message: Option<&str>,
    black_chat_message: Option<&str>,
) -> Result<(Vec<(usize, RgbaImage)>, u128), image::ImageError> {
    let started_at = Instant::now();
    let rendered_tiles: Result<Vec<(usize, RgbaImage)>, image::ImageError> = (0..TILE_COUNT)
        .into_par_iter()
        .map(|tile_index| {
            let tile = render_board_tile_rgba(
                board,
                tile_index,
                RAW_TILE_SQUARE_SIZE,
                white_player_name,
                black_player_name,
                white_chat_message,
                black_chat_message,
            )?;
            Ok((tile_index, tile))
        })
        .collect();

    Ok((rendered_tiles?, started_at.elapsed().as_millis()))
}

fn compose_board_square(rendered_tiles: Vec<(usize, RgbaImage)>) -> RgbaImage {
    let mut board_square = RgbaImage::new(
        RAW_TILE_SIDE * TILE_GRID_DIMENSION,
        RAW_TILE_SIDE * TILE_GRID_DIMENSION,
    );
    for (tile_index, tile) in rendered_tiles {
        let (tile_file, tile_rank_top) = tile_coords(tile_index);
        overlay(
            &mut board_square,
            &tile,
            i64::from(tile_file * RAW_TILE_SIDE),
            i64::from(tile_rank_top * RAW_TILE_SIDE),
        );
    }
    board_square
}

fn compose_wall_backdrop_with_shadow(wall_columns: u32, wall_rows: u32) -> RgbaImage {
    let (wall_frame_width, wall_frame_height) = wall_frame_dimensions(wall_columns, wall_rows);
    let mut wall_backdrop = (*cached_wall_background(wall_columns, wall_rows)).clone();
    let board_w = wall_frame_width.min(wall_frame_height);
    let board_h = board_w;
    let board_x = (wall_frame_width.saturating_sub(board_w)) / 2;
    let board_y = (wall_frame_height.saturating_sub(board_h)) / 2;
    let shadow_offset = 18;
    paint_rect_rgba(
        &mut wall_backdrop,
        board_x.saturating_add(shadow_offset),
        board_y.saturating_add(shadow_offset),
        board_w.saturating_sub(shadow_offset),
        board_h.saturating_sub(shadow_offset),
        Rgba([0, 0, 0, 78]),
    );
    wall_backdrop
}

fn compose_wall_frame_cpu(rendered_tiles: Vec<(usize, RgbaImage)>, wall_columns: u32, wall_rows: u32) -> RgbaImage {
    let (wall_frame_width, wall_frame_height) = wall_frame_dimensions(wall_columns, wall_rows);
    let board_square = compose_board_square(rendered_tiles);
    let board_side = wall_frame_width.min(wall_frame_height);
    let board_square = if board_square.width() != board_side || board_square.height() != board_side
    {
        resize(&board_square, board_side, board_side, FilterType::CatmullRom)
    } else {
        board_square
    };
    let mut wall_frame = compose_wall_backdrop_with_shadow(wall_columns, wall_rows);
    let board_x = (wall_frame_width.saturating_sub(board_square.width())) / 2;
    let board_y = (wall_frame_height.saturating_sub(board_square.height())) / 2;
    overlay(
        &mut wall_frame,
        &board_square,
        i64::from(board_x),
        i64::from(board_y),
    );
    paint_rect_rgba(
        &mut wall_frame,
        board_x,
        board_y,
        board_square.width(),
        3,
        Rgba([248, 240, 230, 120]),
    );
    paint_rect_rgba(
        &mut wall_frame,
        board_x,
        board_y.saturating_add(board_square.height().saturating_sub(3)),
        board_square.width(),
        3,
        Rgba([248, 240, 230, 120]),
    );
    paint_rect_rgba(
        &mut wall_frame,
        board_x,
        board_y,
        3,
        board_square.height(),
        Rgba([248, 240, 230, 120]),
    );
    paint_rect_rgba(
        &mut wall_frame,
        board_x.saturating_add(board_square.width().saturating_sub(3)),
        board_y,
        3,
        board_square.height(),
        Rgba([248, 240, 230, 120]),
    );

    wall_frame
}

fn encode_wall_tiles_to_jpeg_cpu(
    wall_frame: &RgbaImage,
    wall_columns: u32,
    output_tile_count: usize,
) -> Result<Vec<Vec<u8>>, image::ImageError> {
    let encoded_tiles: Result<Vec<(usize, Vec<u8>)>, image::ImageError> = (0..output_tile_count)
        .into_par_iter()
        .map(|tile_index| {
            let tile_file = (tile_index as u32) % wall_columns;
            let tile_rank_top = (tile_index as u32) / wall_columns;
            let crop_x = tile_file * OUTPUT_FRAME_WIDTH;
            let crop_y = tile_rank_top * OUTPUT_FRAME_HEIGHT;
            let tile_view = image::imageops::crop_imm(
                wall_frame,
                crop_x,
                crop_y,
                OUTPUT_FRAME_WIDTH,
                OUTPUT_FRAME_HEIGHT,
            )
            .to_image();
            let rgb = DynamicImage::ImageRgba8(tile_view).to_rgb8();
            let jpeg = encode_rgb_to_jpeg(&rgb, DEFAULT_JPEG_QUALITY)?;
            Ok((tile_index, jpeg))
        })
        .collect();

    let mut tile_jpegs = vec![Vec::new(); output_tile_count];
    for (tile_index, jpeg) in encoded_tiles? {
        tile_jpegs[tile_index] = jpeg;
    }
    Ok(tile_jpegs)
}

fn log_render_timings(frame_id: u64, timings: &RenderTimings) {
    println!(
        "render wall frame #{frame_id}: total={}ms tile_render={}ms wall_compose={}ms jpeg_encode={}ms",
        timings.total_ms,
        timings.tile_render_ms,
        timings.wall_compose_ms,
        timings.jpeg_encode_ms
    );
}

pub fn tile_square_labels(tile_index: usize) -> [String; 4] {
    let (tile_file, tile_rank_top) = tile_coords(tile_index);
    let top_left = square_label(
        tile_file * TILE_DIMENSION_SQUARES,
        tile_rank_top * TILE_DIMENSION_SQUARES,
    );
    let top_right = square_label(
        tile_file * TILE_DIMENSION_SQUARES + 1,
        tile_rank_top * TILE_DIMENSION_SQUARES,
    );
    let bottom_left = square_label(
        tile_file * TILE_DIMENSION_SQUARES,
        tile_rank_top * TILE_DIMENSION_SQUARES + 1,
    );
    let bottom_right = square_label(
        tile_file * TILE_DIMENSION_SQUARES + 1,
        tile_rank_top * TILE_DIMENSION_SQUARES + 1,
    );
    [top_left, top_right, bottom_left, bottom_right]
}

pub fn wall_slot_labels(slot_index: usize, output_tile_count: usize) -> [String; 4] {
    if output_tile_count == TILE_COUNT {
        return tile_square_labels(slot_index);
    }

    let (wall_columns, _) = wall_layout(output_tile_count.max(1));
    let col = (slot_index as u32 % wall_columns) + 1;
    let row = (slot_index as u32 / wall_columns) + 1;
    [
        format!("slot{}", slot_index + 1),
        format!("row{row}"),
        format!("col{col}"),
        "wall".to_string(),
    ]
}

pub fn render_board_tile_jpeg(
    board: &Board,
    tile_index: usize,
    square_size_px: u32,
    jpeg_quality: u8,
    white_player_name: Option<&str>,
    black_player_name: Option<&str>,
    white_chat_message: Option<&str>,
    black_chat_message: Option<&str>,
) -> Result<Vec<u8>, image::ImageError> {
    let image = render_board_tile_rgba(
        board,
        tile_index,
        square_size_px,
        white_player_name,
        black_player_name,
        white_chat_message,
        black_chat_message,
    )?;
    let framed_rgb = compose_output_video_frame(&image);
    encode_rgb_to_jpeg(&framed_rgb, jpeg_quality)
}

fn render_board_tile_rgba(
    board: &Board,
    tile_index: usize,
    square_size_px: u32,
    white_player_name: Option<&str>,
    black_player_name: Option<&str>,
    white_chat_message: Option<&str>,
    black_chat_message: Option<&str>,
) -> Result<RgbaImage, image::ImageError> {
    let mut image = (*cached_tile_base_rgba(tile_index, square_size_px)).clone();

    let (tile_file, tile_rank_top) = tile_coords(tile_index);
    let start_file = tile_file * TILE_DIMENSION_SQUARES;
    let start_rank_top = tile_rank_top * TILE_DIMENSION_SQUARES;

    for local_rank_top in 0..TILE_DIMENSION_SQUARES {
        for local_file in 0..TILE_DIMENSION_SQUARES {
            let board_file = start_file + local_file;
            let board_rank_top = start_rank_top + local_rank_top;
            let board_rank = board_rank_from_top(board_rank_top);

            if let Some(piece) = piece_at(board, board_file, board_rank) {
                let scaled = cached_piece_sprite(piece, square_size_px)?;
                let x = i64::from(local_file * square_size_px);
                let y = i64::from(local_rank_top * square_size_px);
                overlay(&mut image, scaled.as_ref(), x, y);

                if piece.role == Role::King {
                    let player_name = match piece.color {
                        Color::White => white_player_name,
                        Color::Black => black_player_name,
                    };
                    if let Some(player_name) = player_name {
                        draw_king_owner_label(
                            &mut image,
                            local_file,
                            local_rank_top,
                            square_size_px,
                            player_name,
                            (board_file + board_rank).is_multiple_of(2),
                        );
                    }
                }
            }
        }
    }

    draw_cross_tile_king_chat_bubble(
        &mut image,
        board,
        start_file,
        start_rank_top,
        square_size_px,
        Color::White,
        white_chat_message,
    );
    draw_cross_tile_king_chat_bubble(
        &mut image,
        board,
        start_file,
        start_rank_top,
        square_size_px,
        Color::Black,
        black_chat_message,
    );

    Ok(image)
}

fn cached_piece_sprite(
    piece: shakmaty::Piece,
    square_size_px: u32,
) -> Result<Arc<RgbaImage>, image::ImageError> {
    let key = PieceSpriteCacheKey {
        piece_index: piece_index(piece),
        square_size_px,
    };
    let cache = PIECE_SPRITE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(existing) = cache.lock().expect("piece sprite cache poisoned").get(&key) {
        return Ok(existing.clone());
    }

    let decoded = image::load_from_memory(piece_sprite_bytes(piece))?.to_rgba8();
    let scaled = if decoded.width() == square_size_px && decoded.height() == square_size_px {
        decoded
    } else {
        resize(
            &decoded,
            square_size_px,
            square_size_px,
            FilterType::CatmullRom,
        )
    };
    let scaled = Arc::new(scaled);

    let mut guard = cache.lock().expect("piece sprite cache poisoned");
    let entry = guard.entry(key).or_insert_with(|| scaled.clone());
    Ok(entry.clone())
}

fn cached_tile_base_rgba(tile_index: usize, square_size_px: u32) -> Arc<RgbaImage> {
    let key = TileBaseCacheKey {
        tile_index,
        square_size_px,
    };
    let cache = TILE_BASE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(existing) = cache.lock().expect("tile base cache poisoned").get(&key) {
        return existing.clone();
    }

    let built = Arc::new(build_tile_base_rgba(tile_index, square_size_px));
    let mut guard = cache.lock().expect("tile base cache poisoned");
    let entry = guard.entry(key).or_insert_with(|| built.clone());
    entry.clone()
}

fn build_tile_base_rgba(tile_index: usize, square_size_px: u32) -> RgbaImage {
    let tile_side = TILE_DIMENSION_SQUARES * square_size_px;
    let mut image = RgbaImage::new(tile_side, tile_side);
    let (tile_file, tile_rank_top) = tile_coords(tile_index);
    let start_file = tile_file * TILE_DIMENSION_SQUARES;
    let start_rank_top = tile_rank_top * TILE_DIMENSION_SQUARES;

    for local_rank_top in 0..TILE_DIMENSION_SQUARES {
        for local_file in 0..TILE_DIMENSION_SQUARES {
            let board_file = start_file + local_file;
            let board_rank_top = start_rank_top + local_rank_top;
            let board_rank = board_rank_from_top(board_rank_top);
            let color = square_color(board_file, board_rank);
            paint_square_rgba(
                &mut image,
                local_file,
                local_rank_top,
                square_size_px,
                Rgba([color[0], color[1], color[2], 255]),
            );
            draw_square_coordinate_label(
                &mut image,
                local_file,
                local_rank_top,
                square_size_px,
                board_file,
                board_rank_top,
            );
        }
    }

    image
}

fn tile_coords(tile_index: usize) -> (u32, u32) {
    let tile_index = tile_index as u32;
    let tile_file = tile_index % TILE_GRID_DIMENSION;
    let tile_rank_top = (tile_index / TILE_GRID_DIMENSION) % TILE_GRID_DIMENSION;
    (tile_file, tile_rank_top)
}

fn square_label(file: u32, rank_from_top: u32) -> String {
    let file_char = char::from(b'a' + file as u8);
    let rank_display = board_rank_from_top(rank_from_top) + 1;
    format!("{file_char}{rank_display}")
}

fn board_rank_from_top(rank_from_top: u32) -> u32 {
    BOARD_DIMENSION - 1 - rank_from_top
}

fn square_color(file: u32, rank_from_bottom: u32) -> [u8; 3] {
    if (file + rank_from_bottom) % 2 == 0 {
        DARK_SQUARE
    } else {
        LIGHT_SQUARE
    }
}

fn piece_at(board: &Board, file: u32, rank: u32) -> Option<shakmaty::Piece> {
    let square = Square::from_coords(File::new(file), Rank::new(rank));
    board.piece_at(square)
}

fn piece_sprite(piece: shakmaty::Piece) -> Result<RgbaImage, image::ImageError> {
    Ok(image::load_from_memory(piece_sprite_bytes(piece))?.to_rgba8())
}

fn piece_index(piece: shakmaty::Piece) -> u8 {
    match (piece.color, piece.role) {
        (Color::White, Role::Pawn) => 0,
        (Color::White, Role::Knight) => 1,
        (Color::White, Role::Bishop) => 2,
        (Color::White, Role::Rook) => 3,
        (Color::White, Role::Queen) => 4,
        (Color::White, Role::King) => 5,
        (Color::Black, Role::Pawn) => 6,
        (Color::Black, Role::Knight) => 7,
        (Color::Black, Role::Bishop) => 8,
        (Color::Black, Role::Rook) => 9,
        (Color::Black, Role::Queen) => 10,
        (Color::Black, Role::King) => 11,
    }
}

fn piece_sprite_bytes(piece: shakmaty::Piece) -> &'static [u8] {
    match (piece.color, piece.role) {
        (Color::White, Role::Pawn) => {
            include_bytes!("../assets/pieces/wikipedia/wP.png").as_slice()
        }
        (Color::White, Role::Knight) => {
            include_bytes!("../assets/pieces/wikipedia/wN.png").as_slice()
        }
        (Color::White, Role::Bishop) => {
            include_bytes!("../assets/pieces/wikipedia/wB.png").as_slice()
        }
        (Color::White, Role::Rook) => {
            include_bytes!("../assets/pieces/wikipedia/wR.png").as_slice()
        }
        (Color::White, Role::Queen) => {
            include_bytes!("../assets/pieces/wikipedia/wQ.png").as_slice()
        }
        (Color::White, Role::King) => {
            include_bytes!("../assets/pieces/wikipedia/wK.png").as_slice()
        }
        (Color::Black, Role::Pawn) => {
            include_bytes!("../assets/pieces/wikipedia/bP.png").as_slice()
        }
        (Color::Black, Role::Knight) => {
            include_bytes!("../assets/pieces/wikipedia/bN.png").as_slice()
        }
        (Color::Black, Role::Bishop) => {
            include_bytes!("../assets/pieces/wikipedia/bB.png").as_slice()
        }
        (Color::Black, Role::Rook) => {
            include_bytes!("../assets/pieces/wikipedia/bR.png").as_slice()
        }
        (Color::Black, Role::Queen) => {
            include_bytes!("../assets/pieces/wikipedia/bQ.png").as_slice()
        }
        (Color::Black, Role::King) => {
            include_bytes!("../assets/pieces/wikipedia/bK.png").as_slice()
        }
    }
}

fn draw_king_owner_label(
    image: &mut RgbaImage,
    local_file: u32,
    local_rank: u32,
    square_size_px: u32,
    player_name: &str,
    is_dark_square: bool,
) {
    let label = compact_player_label(player_name);
    if label.is_empty() {
        return;
    }

    let text_color = if is_dark_square {
        Rgba([250, 250, 250, 255])
    } else {
        Rgba([20, 20, 20, 255])
    };
    let bg_color = if is_dark_square {
        Rgba([0, 0, 0, 150])
    } else {
        Rgba([255, 255, 255, 170])
    };

    let scale = (square_size_px / 56).max(1);
    let padding = (square_size_px / 40).max(2);
    let text_spacing = scale;
    let glyph_width = 3 * scale;
    let glyph_height = 5 * scale;
    let char_count = label.chars().count() as u32;
    let text_width = char_count
        .saturating_mul(glyph_width)
        .saturating_add(char_count.saturating_sub(1).saturating_mul(text_spacing));

    let square_origin_x = local_file * square_size_px;
    let square_origin_y = local_rank * square_size_px;
    let text_x = square_origin_x + padding;
    let text_y = square_origin_y + square_size_px - glyph_height - padding;

    paint_rect_rgba(
        image,
        text_x.saturating_sub(1),
        text_y.saturating_sub(1),
        text_width + 2,
        glyph_height + 2,
        bg_color,
    );
    draw_bitmap_text_3x5(
        image,
        text_x,
        text_y,
        &label,
        scale,
        text_spacing,
        text_color,
    );
}

fn draw_cross_tile_king_chat_bubble(
    image: &mut RgbaImage,
    board: &Board,
    tile_start_file: u32,
    tile_start_rank_top: u32,
    square_size_px: u32,
    king_color: Color,
    chat_message: Option<&str>,
) {
    let Some(chat_message) = chat_message else {
        return;
    };
    let label = compact_chat_label(chat_message);
    if label.is_empty() {
        return;
    }
    let Some(king_square) = board.king_of(king_color) else {
        return;
    };
    let king_file = u32::from(king_square.file());
    let king_rank = u32::from(king_square.rank());
    let king_rank_top = board_rank_from_top(king_rank);
    let is_dark_square = (king_file + king_rank).is_multiple_of(2);

    let text_color = if is_dark_square {
        Rgba([20, 20, 20, 255])
    } else {
        Rgba([245, 245, 245, 255])
    };
    let bg_color = if is_dark_square {
        Rgba([245, 245, 245, 220])
    } else {
        Rgba([15, 15, 15, 210])
    };

    let scale = (square_size_px / 56).max(1);
    let padding = (square_size_px / 34).max(2);
    let text_spacing = scale;
    let glyph_width = 3 * scale;
    let glyph_height = 5 * scale;
    let char_count = label.chars().count() as u32;
    let text_width = char_count
        .saturating_mul(glyph_width)
        .saturating_add(char_count.saturating_sub(1).saturating_mul(text_spacing));
    let bubble_width = text_width + padding * 2;
    let bubble_height = glyph_height + padding * 2;

    let square_origin_x = king_file * square_size_px;
    let square_origin_y = king_rank_top * square_size_px;

    let bubble_x_global = if king_file == 0 {
        square_origin_x + square_size_px / 2
    } else {
        square_origin_x.saturating_sub(bubble_width / 3)
    };
    let bubble_y_global = square_origin_y + padding;

    let tile_origin_x = tile_start_file * square_size_px;
    let tile_origin_y = tile_start_rank_top * square_size_px;
    let local_bubble_x = bubble_x_global as i32 - tile_origin_x as i32;
    let local_bubble_y = bubble_y_global as i32 - tile_origin_y as i32;
    let local_bubble_right = local_bubble_x + bubble_width as i32;
    let local_bubble_bottom = local_bubble_y + bubble_height as i32;
    let tile_w = image.width() as i32;
    let tile_h = image.height() as i32;

    if local_bubble_right <= 0
        || local_bubble_bottom <= 0
        || local_bubble_x >= tile_w
        || local_bubble_y >= tile_h
    {
        return;
    }

    paint_rect_rgba_clipped(
        image,
        local_bubble_x,
        local_bubble_y,
        bubble_width,
        bubble_height,
        bg_color,
    );
    draw_bitmap_text_3x5_clipped(
        image,
        local_bubble_x + padding as i32,
        local_bubble_y + padding as i32,
        &label,
        scale,
        text_spacing,
        text_color,
    );
}

fn compact_player_label(name: &str) -> String {
    name.trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == ' ' || *ch == '-' || *ch == '_')
        .take(12)
        .collect::<String>()
}

fn compact_chat_label(message: &str) -> String {
    let compact = message
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric()
                || matches!(ch, ' ' | '-' | '_' | '.' | ',' | '!' | '?' | '\'')
            {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>();

    compact
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(160)
        .collect::<String>()
}

fn draw_square_coordinate_label(
    image: &mut RgbaImage,
    local_file: u32,
    local_rank: u32,
    square_size_px: u32,
    board_file: u32,
    board_rank_top: u32,
) {
    let label = square_label(board_file, board_rank_top);
    let rank_from_bottom = board_rank_from_top(board_rank_top);
    let is_dark_square = (board_file + rank_from_bottom) % 2 == 0;
    let text_color = if is_dark_square {
        Rgba([240, 240, 240, 255])
    } else {
        Rgba([24, 24, 24, 255])
    };
    let bg_color = if is_dark_square {
        Rgba([0, 0, 0, 140])
    } else {
        Rgba([255, 255, 255, 160])
    };

    let scale = (square_size_px / 48).max(1);
    let padding = (square_size_px / 32).max(2);
    let text_spacing = scale;
    let glyph_width = 3 * scale;
    let glyph_height = 5 * scale;
    let char_count = label.chars().count() as u32;
    let text_width = char_count
        .saturating_mul(glyph_width)
        .saturating_add(char_count.saturating_sub(1).saturating_mul(text_spacing));

    let square_origin_x = local_file * square_size_px;
    let square_origin_y = local_rank * square_size_px;
    let text_x = square_origin_x + padding;
    let text_y = square_origin_y + padding;

    paint_rect_rgba(
        image,
        text_x.saturating_sub(1),
        text_y.saturating_sub(1),
        text_width + 2,
        glyph_height + 2,
        bg_color,
    );
    draw_bitmap_text_3x5(
        image,
        text_x,
        text_y,
        &label,
        scale,
        text_spacing,
        text_color,
    );
}

fn draw_bitmap_text_3x5(
    image: &mut RgbaImage,
    x: u32,
    y: u32,
    text: &str,
    scale: u32,
    spacing: u32,
    color: Rgba<u8>,
) {
    let mut cursor_x = x;
    for ch in text.chars() {
        if let Some(glyph) = glyph_3x5(ch.to_ascii_lowercase()) {
            draw_glyph_3x5(image, cursor_x, y, scale, color, glyph);
        }
        cursor_x += 3 * scale + spacing;
    }
}

fn draw_bitmap_text_3x5_clipped(
    image: &mut RgbaImage,
    x: i32,
    y: i32,
    text: &str,
    scale: u32,
    spacing: u32,
    color: Rgba<u8>,
) {
    let mut cursor_x = x;
    for ch in text.chars() {
        if let Some(glyph) = glyph_3x5(ch.to_ascii_lowercase()) {
            draw_glyph_3x5_clipped(image, cursor_x, y, scale, color, glyph);
        }
        cursor_x += (3 * scale + spacing) as i32;
    }
}

fn draw_glyph_3x5(
    image: &mut RgbaImage,
    x: u32,
    y: u32,
    scale: u32,
    color: Rgba<u8>,
    rows: [u8; 5],
) {
    for (row_idx, row_bits) in rows.into_iter().enumerate() {
        for col in 0..3 {
            let bit_mask = 1 << (2 - col);
            if row_bits & bit_mask == 0 {
                continue;
            }
            paint_rect_rgba(
                image,
                x + (col as u32) * scale,
                y + (row_idx as u32) * scale,
                scale,
                scale,
                color,
            );
        }
    }
}

fn draw_glyph_3x5_clipped(
    image: &mut RgbaImage,
    x: i32,
    y: i32,
    scale: u32,
    color: Rgba<u8>,
    rows: [u8; 5],
) {
    for (row_idx, row_bits) in rows.into_iter().enumerate() {
        for col in 0..3_u32 {
            let bit_mask = 1_u8 << (2 - col);
            if row_bits & bit_mask == 0 {
                continue;
            }
            paint_rect_rgba_clipped(
                image,
                x + (col * scale) as i32,
                y + (row_idx as u32 * scale) as i32,
                scale,
                scale,
                color,
            );
        }
    }
}

fn glyph_3x5(ch: char) -> Option<[u8; 5]> {
    Some(match ch {
        ' ' => [0b000, 0b000, 0b000, 0b000, 0b000],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '_' => [0b000, 0b000, 0b000, 0b000, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ',' => [0b000, 0b000, 0b000, 0b010, 0b100],
        '!' => [0b010, 0b010, 0b010, 0b000, 0b010],
        '?' => [0b110, 0b001, 0b010, 0b000, 0b010],
        '\'' => [0b010, 0b010, 0b000, 0b000, 0b000],
        'a' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'b' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'c' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'd' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'e' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'f' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'g' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'h' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'i' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'j' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'k' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'l' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'm' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'n' => [0b101, 0b111, 0b111, 0b111, 0b101],
        'o' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'p' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'q' => [0b111, 0b101, 0b101, 0b111, 0b001],
        'r' => [0b110, 0b101, 0b110, 0b101, 0b101],
        's' => [0b011, 0b100, 0b010, 0b001, 0b110],
        't' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'u' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'v' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'w' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'x' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b110, 0b001, 0b010, 0b100, 0b111],
        '3' => [0b110, 0b001, 0b010, 0b001, 0b110],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b110, 0b001, 0b110],
        '6' => [0b011, 0b100, 0b110, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b110],
        _ => return None,
    })
}

fn paint_rect_rgba(
    image: &mut RgbaImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: Rgba<u8>,
) {
    let x_end = x.saturating_add(width).min(image.width());
    let y_end = y.saturating_add(height).min(image.height());
    for py in y..y_end {
        for px in x..x_end {
            image.put_pixel(px, py, color);
        }
    }
}

fn paint_rect_rgba_clipped(
    image: &mut RgbaImage,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    color: Rgba<u8>,
) {
    let x_start = x.max(0) as u32;
    let y_start = y.max(0) as u32;
    let x_end = (x + width as i32).min(image.width() as i32).max(0) as u32;
    let y_end = (y + height as i32).min(image.height() as i32).max(0) as u32;

    if x_start >= x_end || y_start >= y_end {
        return;
    }

    for py in y_start..y_end {
        for px in x_start..x_end {
            image.put_pixel(px, py, color);
        }
    }
}

fn paint_square_rgb(
    image: &mut RgbImage,
    file: u32,
    rank: u32,
    square_size_px: u32,
    color: Rgb<u8>,
) {
    let start_x = file * square_size_px;
    let start_y = rank * square_size_px;

    for y in start_y..start_y + square_size_px {
        for x in start_x..start_x + square_size_px {
            image.put_pixel(x, y, color);
        }
    }
}

fn paint_square_rgba(
    image: &mut RgbaImage,
    file: u32,
    rank: u32,
    square_size_px: u32,
    color: Rgba<u8>,
) {
    let start_x = file * square_size_px;
    let start_y = rank * square_size_px;

    for y in start_y..start_y + square_size_px {
        for x in start_x..start_x + square_size_px {
            image.put_pixel(x, y, color);
        }
    }
}

fn paint_wall_background(image: &mut RgbaImage) {
    let width = image.width().max(1);
    let height = image.height().max(1);
    for y in 0..height {
        for x in 0..width {
            let center_dist_x = (x as i64 - (width as i64 / 2)).unsigned_abs() as u32;
            let edge_factor = center_dist_x.saturating_mul(200) / (width / 2).max(1);
            let vertical_soften = (y.saturating_mul(70)) / height;

            let base_r = 58_u32
                .saturating_sub(edge_factor / 8)
                .saturating_sub(vertical_soften / 3);
            let base_g = 44_u32
                .saturating_sub(edge_factor / 9)
                .saturating_sub(vertical_soften / 3);
            let base_b = 33_u32
                .saturating_sub(edge_factor / 10)
                .saturating_sub(vertical_soften / 4);

            image.put_pixel(
                x,
                y,
                Rgba([
                    base_r.clamp(16, 72) as u8,
                    base_g.clamp(12, 56) as u8,
                    base_b.clamp(10, 44) as u8,
                    255,
                ]),
            );
        }
    }
}

fn compose_output_video_frame(tile: &RgbaImage) -> RgbImage {
    let mut frame = (*cached_output_frame_background()).clone();

    let panel_side = OUTPUT_FRAME_HEIGHT.min(tile.width().min(tile.height()));
    let board_panel = if tile.width() == panel_side && tile.height() == panel_side {
        tile.clone()
    } else {
        resize(tile, panel_side, panel_side, FilterType::CatmullRom)
    };
    let panel_x = (OUTPUT_FRAME_WIDTH.saturating_sub(panel_side)) / 2;
    let panel_y = (OUTPUT_FRAME_HEIGHT.saturating_sub(panel_side)) / 2;
    let shadow_offset = 6;
    paint_rect_rgba(
        &mut frame,
        panel_x.saturating_add(shadow_offset),
        panel_y.saturating_add(shadow_offset),
        panel_side,
        panel_side,
        Rgba([0, 0, 0, 72]),
    );
    overlay(
        &mut frame,
        &board_panel,
        i64::from(panel_x),
        i64::from(panel_y),
    );

    let border = Rgba([244, 236, 226, 110]);
    paint_rect_rgba(&mut frame, panel_x, panel_y, panel_side, 2, border);
    paint_rect_rgba(
        &mut frame,
        panel_x,
        panel_y.saturating_add(panel_side.saturating_sub(2)),
        panel_side,
        2,
        border,
    );
    paint_rect_rgba(&mut frame, panel_x, panel_y, 2, panel_side, border);
    paint_rect_rgba(
        &mut frame,
        panel_x.saturating_add(panel_side.saturating_sub(2)),
        panel_y,
        2,
        panel_side,
        border,
    );

    DynamicImage::ImageRgba8(frame).to_rgb8()
}

fn cached_wall_background(wall_columns: u32, wall_rows: u32) -> Arc<RgbaImage> {
    let key = WallLayoutKey {
        columns: wall_columns.max(1),
        rows: wall_rows.max(1),
    };
    let cache = WALL_BACKGROUND_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(existing) = cache.lock().expect("wall background cache poisoned").get(&key) {
        return existing.clone();
    }

    let (wall_frame_width, wall_frame_height) = wall_frame_dimensions(key.columns, key.rows);
    let mut image = RgbaImage::new(wall_frame_width, wall_frame_height);
    paint_wall_background(&mut image);
    let image = Arc::new(image);

    let mut guard = cache.lock().expect("wall background cache poisoned");
    let entry = guard.entry(key).or_insert_with(|| image.clone());
    entry.clone()
}

fn cached_output_frame_background() -> Arc<RgbaImage> {
    OUTPUT_FRAME_BACKGROUND_CACHE
        .get_or_init(|| {
            let mut image = RgbaImage::new(OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT);
            paint_wall_background(&mut image);
            Arc::new(image)
        })
        .clone()
}

fn encode_rgb_to_jpeg(image: &RgbImage, jpeg_quality: u8) -> Result<Vec<u8>, image::ImageError> {
    let mut jpeg = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg, jpeg_quality);
    encoder.encode(
        image.as_raw(),
        image.width(),
        image.height(),
        ColorType::Rgb8.into(),
    )?;
    Ok(jpeg)
}
