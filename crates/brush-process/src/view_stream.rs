use crate::message::ProcessMessage;

use std::{pin::pin, sync::Arc};

use async_fn_stream::TryStreamEmitter;
use brush_serde::{self, Deformations};
use brush_vfs::BrushVfs;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use tokio::io::AsyncReadExt;
use tokio_stream::StreamExt;

// try to find and load deformation file for a ply
async fn try_load_deformations(vfs: &BrushVfs) -> Option<Deformations> {
    let bin_path = vfs.file_paths().find(|p| p.extension().is_some_and(|e| e == "bin"))?;
    let mut reader = vfs.reader_at_path(&bin_path).await.ok()?;
    let mut data = Vec::new();
    reader.read_to_end(&mut data).await.ok()?;

    if !brush_serde::is_deformation_file(&data) {
        return None;
    }

    let deforms = Deformations::from_bytes(&data).ok()?;
    log::info!(
        "Loaded deformation file {:?}: {} frames, {} points",
        bin_path,
        deforms.num_frames,
        deforms.num_points
    );
    Some(deforms)
}

pub(crate) async fn view_stream(
    vfs: Arc<BrushVfs>,
    device: WgpuDevice,
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
) -> anyhow::Result<()> {
    let mut ply_paths: Vec<_> = vfs
        .file_paths()
        .filter(|p| p.extension().is_some_and(|e| e == "ply"))
        .collect();
    alphanumeric_sort::sort_path_slice(&mut ply_paths);
    let client = WgpuRuntime::client(&device);

    if ply_paths.len() == 1 {
        let ply_path = &ply_paths[0];

        if let Some(deformations) = try_load_deformations(&vfs).await {
            log::info!("Loading single ply file");

            emitter
                .emit(ProcessMessage::StartLoading { training: false })
                .await;

            let base_splat =
                brush_serde::load_splat_from_ply(vfs.reader_at_path(ply_path).await?, None).await?;

            log::info!(
                "Streaming {} frames",
                deformations.num_frames
            );

            let mut anim_stream = pin!(brush_serde::stream_animated_splats(base_splat, deformations));

            while let Some(message) = anim_stream.next().await {
                let message = message?;

                let splats = message.data.into_splats(&device);

                client.memory_cleanup();

                emitter
                    .emit(ProcessMessage::ViewSplats {
                        up_axis: message.meta.up_axis,
                        splats: Box::new(splats),
                        frame: message.meta.current_frame,
                        total_frames: message.meta.frame_count,
                        progress: message.meta.progress,
                    })
                    .await;
            }

            emitter.emit(ProcessMessage::DoneLoading).await;
            return Ok(());
        }
    }


    for (i, path) in ply_paths.iter().enumerate() {
        log::info!("Loading ply file: {:?}", path);

        emitter
            .emit(ProcessMessage::StartLoading { training: false })
            .await;

        let mut splat_stream = pin!(brush_serde::stream_splat_from_ply(
            vfs.reader_at_path(path).await?,
            None,
            true,
        ));

        while let Some(message) = splat_stream.next().await {
            let message = message?;

            // Convert SplatData to Splats using simple defaults
            let splats = message.data.into_splats(&device);

            // If there's multiple ply files in a zip, don't support animated plys, that would
            // get rather mind bending.
            let (frame, total_frames) = if ply_paths.len() == 1 {
                (message.meta.current_frame, message.meta.frame_count)
            } else {
                (i as u32, ply_paths.len() as u32)
            };

            // As loading concatenates splats each time, memory usage tends to accumulate a lot
            // over time. Clear out memory after each step to prevent this buildup.
            client.memory_cleanup();

            emitter
                .emit(ProcessMessage::ViewSplats {
                    up_axis: message.meta.up_axis,
                    splats: Box::new(splats),
                    frame,
                    total_frames,
                    progress: message.meta.progress,
                })
                .await;
        }
    }

    emitter.emit(ProcessMessage::DoneLoading).await;

    Ok(())
}
