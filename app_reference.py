# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
from ultralytics import YOLO

MODEL_CHOICES = [
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26n-seg",
    "yolo26s-seg",
    "yolo26m-seg",
    "yolo26n-pose",
    "yolo26s-pose",
    "yolo26m-pose",
    "yolo26n-obb",
    "yolo26s-obb",
    "yolo26m-obb",
    "yolo26n-cls",
    "yolo26s-cls",
    "yolo26m-cls",
]

IMAGE_SIZE_CHOICES = [320, 640, 1024]
CUSTOM_CSS = (Path(__file__).parent / "ultralytics.css").read_text()


def predict_image(img, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in an image using a Ultralytics YOLO model with adjustable confidence and IOU thresholds."""
    model = YOLO(model_name)
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    for r in results:
        im_array = r.plot(labels=show_labels, conf=show_conf)
        im = Image.fromarray(im_array[..., ::-1])

    return im


def predict_video(video_path, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in a video using a Ultralytics YOLO model and returns the annotated video."""
    if video_path is None:
        return None

    model = YOLO(model_name)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_output.name
    temp_output.close()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        # Get the annotated frame
        annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
        out.write(annotated_frame)

    cap.release()
    out.release()

    return output_path


# Cache model for streaming performance
_model_cache = {}


def get_model(model_name):
    """Get or create a cached model instance."""
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(model_name)
    return _model_cache[model_name]


def predict_webcam(frame, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in a webcam frame using a Ultralytics YOLO model (optimized for streaming)."""
    if frame is None:
        return None

    # Use cached model for better streaming performance
    model = get_model(model_name)

    if isinstance(frame, np.ndarray):
        # Gradio webcam sends RGB, but Ultralytics YOLO expects BGR for OpenCV operations
        # Convert RGB to BGR for YOLO
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run inference
        results = model.predict(
            source=frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        # YOLO's plot() returns BGR, convert back to RGB for Gradio display
        annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
        # Convert BGR to RGB for Gradio
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    return None


# Create the Gradio app with tabs
with gr.Blocks(title="Ultralytics YOLO26 Inference 🚀") as demo:
    gr.Markdown(
        """
<div align="center">
  <p>
    <a href="https://platform.ultralytics.com/?utm_source=huggingface&utm_medium=referral&utm_campaign=yolo26&utm_content=banner" target="_blank">
      <img width="50%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>
  <p style="margin: 3px 0;">
    <a href="https://docs.ultralytics.com/zh/">中文</a> | <a href="https://docs.ultralytics.com/ko/">한국어</a> | <a href="https://docs.ultralytics.com/ja/">日本語</a> | <a href="https://docs.ultralytics.com/ru/">Русский</a> | <a href="https://docs.ultralytics.com/de/">Deutsch</a> | <a href="https://docs.ultralytics.com/fr/">Français</a> | <a href="https://docs.ultralytics.com/es">Español</a> | <a href="https://docs.ultralytics.com/pt/">Português</a> | <a href="https://docs.ultralytics.com/tr/">Türkçe</a> | <a href="https://docs.ultralytics.com/vi/">Tiếng Việt</a> | <a href="https://docs.ultralytics.com/ar/">العربية</a>
  </p>
  <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 3px; margin-top: 3px;">
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://pepy.tech/projects/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Ultralytics YOLO Citation"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
  </div>
  <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 3px; margin-top: 3px;">
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo26"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
  </div>
</div>
[Ultralytics]( https://www.ultralytics.com/?utm_source=huggingface&utm_medium=referral&utm_campaign=yolo26&utm_content=contextual) [YOLO26](https://platform.ultralytics.com/ultralytics/yolo26?utm_source=huggingface&utm_medium=referral&utm_campaign=yolo26&utm_content=contextual_model_link) is the latest evolution in the YOLO series of real-time object detectors, engineered from the ground up for edge and low-power devices. It introduces a streamlined design that removes unnecessary complexity while integrating targeted innovations to deliver faster, lighter, and more accessible deployment.
"""
    )

    with gr.Tabs():
        # Image Tab
        with gr.TabItem("📷 Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    img_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    img_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolo26n")
                    img_labels = gr.Checkbox(value=True, label="Show Labels")
                    img_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    img_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    img_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    img_output = gr.Image(type="pil", label="Result")

            img_btn.click(
                predict_image,
                inputs=[img_input, img_conf, img_iou, img_model, img_labels, img_conf_show, img_size],
                outputs=img_output,
            )

            gr.Examples(
                examples=[
                    ["https://ultralytics.com/images/bus.jpg", 0.25, 0.7, "yolo26n", True, True, 640],
                    ["https://ultralytics.com/images/zidane.jpg", 0.25, 0.7, "yolo26n-seg", True, True, 640],
                    ["https://ultralytics.com/images/boats.jpg", 0.25, 0.7, "yolo26n-obb", True, True, 1024],
                ],
                inputs=[img_input, img_conf, img_iou, img_model, img_labels, img_conf_show, img_size],
            )

        # Video Tab
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    vid_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    vid_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    vid_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolo26n")
                    vid_labels = gr.Checkbox(value=True, label="Show Labels")
                    vid_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    vid_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Result")

            vid_btn.click(
                predict_video,
                inputs=[vid_input, vid_conf, vid_iou, vid_model, vid_labels, vid_conf_show, vid_size],
                outputs=vid_output,
            )

        # Webcam Tab - Real-time streaming
        with gr.TabItem("📹 Webcam"):
            gr.Markdown("### Real-time Webcam Detection")
            gr.Markdown("Enable streaming for live detection as you move!")
            with gr.Row():
                with gr.Column():
                    webcam_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    webcam_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    webcam_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolo26n")
                    webcam_labels = gr.Checkbox(value=True, label="Show Labels")
                    webcam_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    webcam_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                with gr.Column():
                    # Streaming webcam input with real-time output
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam (streaming)",
                        streaming=True,
                    )
                    webcam_output = gr.Image(type="numpy", label="Detection Result")

            # Stream event for real-time detection
            webcam_input.stream(
                predict_webcam,
                inputs=[
                    webcam_input,
                    webcam_conf,
                    webcam_iou,
                    webcam_model,
                    webcam_labels,
                    webcam_conf_show,
                    webcam_size,
                ],
                outputs=webcam_output,
            )

demo.launch(css=CUSTOM_CSS, ssr_mode=False)
