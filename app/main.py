from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import DEFAULT_PROCESSING_CONFIG, SUPPORTED_VIDEO_EXTENSIONS, UPLOAD_DIR, ensure_runtime_dirs
from app.schemas import ProcessingConfig
from app.services.jobs import JobManager

app = FastAPI(title="Crowd Clustering Detection Dashboard", version="1.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
job_manager = JobManager()

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.on_event("startup")
def startup_event() -> None:
    ensure_runtime_dirs()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "defaults": DEFAULT_PROCESSING_CONFIG.to_dict(),
            "supported_formats": sorted(extension.replace(".", "").upper() for extension in SUPPORTED_VIDEO_EXTENSIONS),
        },
    )


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    detector_model: str = Form(DEFAULT_PROCESSING_CONFIG.model_name, alias="model_name"),
    device: str = Form(""),
    imgsz: int = Form(DEFAULT_PROCESSING_CONFIG.imgsz),
    conf_thresh: float = Form(DEFAULT_PROCESSING_CONFIG.conf_thresh),
    iou_thresh: float = Form(DEFAULT_PROCESSING_CONFIG.iou_thresh),
    stationarity_seconds: float = Form(DEFAULT_PROCESSING_CONFIG.stationarity_seconds),
    stationarity_radius: float = Form(DEFAULT_PROCESSING_CONFIG.stationarity_radius),
    cluster_eps: float = Form(DEFAULT_PROCESSING_CONFIG.cluster_eps),
    min_cluster_size: int = Form(DEFAULT_PROCESSING_CONFIG.min_cluster_size),
    history_seconds: float = Form(DEFAULT_PROCESSING_CONFIG.history_seconds),
    perspective_height_reference: float = Form(DEFAULT_PROCESSING_CONFIG.perspective_height_reference),
    perspective_gain: float = Form(DEFAULT_PROCESSING_CONFIG.perspective_gain),
    trail_length: int = Form(DEFAULT_PROCESSING_CONFIG.trail_length),
    track_buffer: int = Form(DEFAULT_PROCESSING_CONFIG.track_buffer),
    track_high_thresh: float = Form(DEFAULT_PROCESSING_CONFIG.track_high_thresh),
    track_low_thresh: float = Form(DEFAULT_PROCESSING_CONFIG.track_low_thresh),
    new_track_thresh: float = Form(DEFAULT_PROCESSING_CONFIG.new_track_thresh),
    match_thresh: float = Form(DEFAULT_PROCESSING_CONFIG.match_thresh),
) -> JSONResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")

    suffix = Path(video.filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix or 'unknown'}'. Supported formats: MP4, AVI, MOV.",
        )

    config = ProcessingConfig(
        model_name=detector_model,
        device=device,
        imgsz=imgsz,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        stationarity_seconds=stationarity_seconds,
        stationarity_radius=stationarity_radius,
        cluster_eps=cluster_eps,
        min_cluster_size=min_cluster_size,
        history_seconds=history_seconds,
        perspective_height_reference=perspective_height_reference,
        perspective_gain=perspective_gain,
        trail_length=trail_length,
        track_buffer=track_buffer,
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        match_thresh=match_thresh,
    ).sanitized()

    upload_path = UPLOAD_DIR / _build_upload_name(video.filename, suffix)
    with upload_path.open("wb") as file_handle:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            file_handle.write(chunk)
    await video.close()

    job = job_manager.create_job(input_path=upload_path, filename=video.filename, config=config)
    return JSONResponse(status_code=202, content=job.snapshot().to_dict())


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.snapshot().to_dict()


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> dict:
    try:
        job = job_manager.pause_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    return job.snapshot().to_dict()


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> dict:
    try:
        job = job_manager.resume_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    return job.snapshot().to_dict()


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return StreamingResponse(
        job_manager.stream_frames(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "completed" or not job.output_path.exists():
        raise HTTPException(status_code=409, detail="Processed video is not ready yet.")
    return FileResponse(
        path=job.output_path,
        filename=f"annotated_{job.filename.rsplit('.', 1)[0]}.mp4",
        media_type="video/mp4",
    )


def _build_upload_name(filename: str, suffix: str) -> str:
    stem = Path(filename).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "upload"
    return f"{uuid4().hex}_{safe_stem}{suffix}"
