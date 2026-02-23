# Importing the requirements
import warnings

import gradio as gr
from src.yolo.predict_pose import predict_pose, predict_video_pose, predict_webcam_pose

warnings.filterwarnings("ignore")


# Pose-specific model choices
MODEL_CHOICES = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
]

# Image size choices
IMAGE_SIZE_CHOICES = [320, 640, 1024]


# Create the Gradio app with tabs
with gr.Blocks(title="YOLO Pose Estimation") as demo:
    gr.Markdown(
        """
# YOLO Pose Estimation
Gradio Demo for YOLO Pose Estimation models. Detect and predict the poses of people in images, videos, or via webcam.
Supports YOLOv8, YOLO11, and YOLO26 pose models.\n
See [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) | [Models Page](https://docs.ultralytics.com/models/) for more details.
"""
    )

    with gr.Tabs():
        # Image Tab
        with gr.TabItem("📷 Image"):
            with gr.Row():
                with gr.Column():
                    # Options for image input
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_conf = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.25,
                        label="Confidence threshold",
                    )
                    img_iou = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.45,
                        label="IoU threshold",
                    )
                    img_model = gr.Radio(
                        choices=MODEL_CHOICES,
                        label="Model Name",
                        value="yolo26n-pose.pt",
                    )
                    img_labels = gr.Checkbox(value=True, label="Show Labels")
                    img_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    img_size = gr.Radio(
                        choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640
                    )
                    img_btn = gr.Button("Detect Poses", variant="primary")
                with gr.Column():
                    img_output = gr.Image(type="pil", label="Result")

            # Button to predict the poses
            img_btn.click(
                predict_pose,
                inputs=[
                    img_input,
                    img_conf,
                    img_iou,
                    img_model,
                    img_labels,
                    img_conf_show,
                    img_size,
                ],
                outputs=img_output,
            )

            # Examples to predict the poses
            gr.Examples(
                examples=[
                    [
                        "images/posing-sample-image1.jpg",
                        0.25,
                        0.45,
                        "yolo26n-pose.pt",
                        True,
                        True,
                        640,
                    ],
                    [
                        "images/posing-sample-image2.png",
                        0.25,
                        0.45,
                        "yolo11n-pose.pt",
                        True,
                        True,
                        640,
                    ],
                    [
                        "images/posing-sample-image3.png",
                        0.25,
                        0.45,
                        "yolov8n-pose.pt",
                        True,
                        True,
                        640,
                    ],
                ],
                inputs=[
                    img_input,
                    img_conf,
                    img_iou,
                    img_model,
                    img_labels,
                    img_conf_show,
                    img_size,
                ],
            )

        # Video Tab
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    # Options for video input
                    vid_input = gr.Video(label="Upload Video")
                    vid_conf = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.25,
                        label="Confidence threshold",
                    )
                    vid_iou = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.45,
                        label="IoU threshold",
                    )
                    vid_model = gr.Radio(
                        choices=MODEL_CHOICES,
                        label="Model Name",
                        value="yolo26n-pose.pt",
                    )
                    vid_labels = gr.Checkbox(value=True, label="Show Labels")
                    vid_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    vid_size = gr.Radio(
                        choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640
                    )
                    vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Result")

            # Button to predict the poses
            vid_btn.click(
                predict_video_pose,
                inputs=[
                    vid_input,
                    vid_conf,
                    vid_iou,
                    vid_model,
                    vid_labels,
                    vid_conf_show,
                    vid_size,
                ],
                outputs=vid_output,
            )

        # Webcam Tab - Real-time streaming
        with gr.TabItem("📹 Webcam"):
            gr.Markdown("### Real-time Webcam Pose Detection")
            gr.Markdown("Enable streaming for live pose detection as you move!")
            with gr.Row():
                with gr.Column():
                    # Options for webcam input
                    webcam_conf = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.25,
                        label="Confidence threshold",
                    )
                    webcam_iou = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.45,
                        label="IoU threshold",
                    )
                    webcam_model = gr.Radio(
                        choices=MODEL_CHOICES,
                        label="Model Name",
                        value="yolo26n-pose.pt",
                    )
                    webcam_labels = gr.Checkbox(value=True, label="Show Labels")
                    webcam_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    webcam_size = gr.Radio(
                        choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640
                    )
                with gr.Column():
                    # Streaming webcam input with real-time output
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam (streaming)",
                        streaming=True,
                    )
                    webcam_output = gr.Image(type="numpy", label="Detection Result")

            # Stream the webcam input to predict the poses
            webcam_input.stream(
                predict_webcam_pose,
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

# Launch the app
demo.launch(debug=False, theme=gr.themes.Base())
