import sys
import tempfile
import threading
from typing import Optional

import PIL.Image as Image
import numpy as np

import cv2
import gradio as gr
from src.exception import CustomExceptionHandling
from src.logger import logging
from ultralytics import YOLO


def predict_pose(
    img: Optional[str],
    conf_threshold: float,
    iou_threshold: float,
    model_name: str,
    show_labels: bool,
    show_conf: bool,
    imgsz: int,
) -> Image.Image:
    """Predicts poses in an image using a YOLO model with adjustable confidence and IOU thresholds."""
    try:
        if img is None:
            raise gr.Error("Please provide an image.")

        model = YOLO(model_name)

        results = model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            device="cpu",
        )

        for r in results:
            im_array = r.plot(labels=show_labels, conf=show_conf)
            # r.plot() returns BGR; reverse channel order to RGB for PIL
            im = Image.fromarray(im_array[..., ::-1])

        logging.info("Pose estimated successfully.")
        return im

    except gr.Error:
        raise
    except Exception as e:
        raise CustomExceptionHandling(e, sys) from e


def predict_video_pose(
    video_path: Optional[str],
    conf_threshold: float,
    iou_threshold: float,
    model_name: str,
    show_labels: bool,
    show_conf: bool,
    imgsz: int,
) -> Optional[str]:
    """Predicts poses in a video using a YOLO model and returns the annotated video path."""
    try:
        if video_path is None:
            raise gr.Error("Please provide a video.")

        model = YOLO(model_name)

        # Open briefly just to read video properties for VideoWriter initialisation
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Write to a temp file so Gradio can serve it without overwriting the source
        temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = temp_output.name
        temp_output.close()

        # mp4v produces an MP4-compatible stream readable by most browsers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # stream=True yields a memory-efficient generator instead of loading
        # all frames into memory at once — recommended for video by the docs
        for r in model.predict(
            source=video_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            stream=True,
            verbose=False,
            device="cpu",
        ):
            # plot() returns a BGR ndarray — write directly without conversion
            out.write(r.plot(labels=show_labels, conf=show_conf))

        # Release handle so the output file is fully flushed before returning
        out.release()

        logging.info("Video pose estimation completed successfully.")
        return output_path

    except gr.Error:
        raise
    except Exception as e:
        raise CustomExceptionHandling(e, sys) from e


# Module-level cache: avoids reloading weights on every streamed frame
_model_cache: dict[str, YOLO] = {}
# Lock ensures only one thread loads a model at a time (concurrent Gradio users)
_model_cache_lock = threading.Lock()


def _get_model(model_name: str) -> YOLO:
    """Return a cached YOLO instance, loading it on first use (thread-safe)."""
    with _model_cache_lock:
        if model_name not in _model_cache:
            _model_cache[model_name] = YOLO(model_name)
        return _model_cache[model_name]


def predict_webcam_pose(
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    model_name: str,
    show_labels: bool,
    show_conf: bool,
    imgsz: int,
) -> Optional[np.ndarray]:
    """Predicts poses in a webcam frame using a YOLO model (optimized for streaming)."""
    try:
        if frame is None:
            return None

        # Use the cached model to avoid per-frame weight loading overhead
        model = _get_model(model_name)

        if isinstance(frame, np.ndarray):
            # Gradio streams frames as RGB; YOLO/OpenCV expect BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model.predict(
                source=frame_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                verbose=False,
                device="cpu",
            )

            # plot() returns BGR; convert back to RGB for Gradio display
            annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
            return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        return None

    except gr.Error:
        raise
    except Exception as e:
        raise CustomExceptionHandling(e, sys) from e
