import gradio as gr
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import utils
from skimage.morphology import skeletonize

executor = ThreadPoolExecutor(max_workers=8)


def change_sync(im, current_PID, current_file_name):
    if not current_file_name:
        return im.get("composite", None)

    base_name = os.path.splitext(current_file_name)[0]
    png_path = f"workspace/{current_PID}/labels_ldc/{base_name}.png"
    drawn_output_path = f"workspace/{current_PID}/drawns/{current_file_name}"

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    os.makedirs(os.path.dirname(drawn_output_path), exist_ok=True)

    check_labeled = False
    layers = im.get("layers", [])
    if len(layers) > 0 and layers[0] is not None:
        layer_0 = layers[0]
    else:
        layer_0 = []

    # layer_0 = im.get("layers", [None])[0]

    # Save label as skeleton
    if isinstance(layer_0, np.ndarray) and layer_0.size > 0:
        try:
            gray = cv2.cvtColor(layer_0, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            skeleton = skeletonize(binary > 0)  # convert to bool before skeletonize
            if np.any(skeleton):
                skeleton_uint8 = (skeleton * 255).astype(np.uint8)
                kernel = np.ones(
                    (3, 3), np.uint8
                )  # 3x3 kernel to slightly thicken lines
                dilated = cv2.dilate(skeleton_uint8, kernel, iterations=1)
                cv2.imwrite(png_path, dilated)
                check_labeled = True
        except Exception as e:
            print("Error saving label:", e)

    # Save composite image with label
    composite = im.get("composite", None)
    if check_labeled and isinstance(composite, np.ndarray) and composite.size > 0:
        try:
            composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
            cv2.imwrite(drawn_output_path, composite_bgr)
        except Exception as e:
            print("Error saving image + label:", e)

    return composite


async def img_change_action(im, current_PID, current_file_name):
    return await asyncio.get_event_loop().run_in_executor(
        executor, change_sync, im, current_PID, current_file_name
    )


def load_next_image(index, current_PID, zip_file):
    _ = utils.handle_zip(current_PID, zip_file)

    image_files = utils.get_images(current_PID)
    if not image_files:
        return None, index, "", list_images(current_PID)
    index = (index + 1) % len(image_files)  # loop back to first image
    file_name = os.path.basename(image_files[index])

    annotated_dir = f"workspace/{current_PID}/labels_ldc"
    os.makedirs(annotated_dir, exist_ok=True)
    return image_files[index], index, file_name, list_images(current_PID)


def list_images(current_PID):
    folder_path = f"workspace/{current_PID}/images"
    valid_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".tiff",
    }  # Add any others if needed

    file_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in valid_exts
    ]
    return file_paths


def show_selected_image(evt: gr.SelectData, im, current_PID):
    file_name = evt.value["image"]["orig_name"]
    path_to_label_folder = f"workspace/{current_PID}/labels_ldc/"
    edge_img = (
        path_to_label_folder + os.path.splitext(os.path.basename(file_name))[0] + ".png"
    )
    f"workspace/{current_PID}/images"

    img_path = f"workspace/{current_PID}/images/{evt.value['image']['orig_name']}"
    drawn_img = f"workspace/{current_PID}/drawns/{evt.value['image']['orig_name']}"
    drawn_img = drawn_img if os.path.exists(drawn_img) else img_path

    edge_img = edge_img if os.path.exists(edge_img) else None

    return img_path, file_name, drawn_img, edge_img


def blank_image():
    return 255 * np.ones((512, 512, 3), dtype=np.uint8)


with gr.Blocks(
    css=".tall-button { height: 85px !important; font-size: 20px; }"
    ".gray-row { background-color: #F1F1F1; padding: 10px; border-radius: 8px; }"
    ".orange-row { background-color: #FFD580 !important; color: white !important; border: 2px solid #FFD580; border-radius: 6px; padding: 8px; }"
) as demo:
    # ========== HTML ============
    gr.Markdown("# 📝 Annotation tools for Edge Detection (LDC) ")
    index_state = gr.State(0)
    coord_dict = gr.State({})

    with gr.Row(variant="compact"):
        zip_input = gr.File(
            file_types=[".zip"],
            label="Upload ZIP File",
            height="80px",
            scale=5,
            elem_classes="orange-row",
        )
        current_PID = gr.Textbox(
            value="", label="Current working PID", scale=3, interactive=True
        )

        Upload_btn = gr.Button(
            "🚀 || UPLOAD IMAGES || or 🔄 || RETRIEVE PREVIOUS WORK BY PID ||",
            variant="huggingface",
            scale=10,
            elem_classes="tall-button",
        )

    with gr.Row(variant="compact"):
        current_file_name = gr.Textbox(
            value="",
            label="Current working file name",
            scale=2,
        )
        gallery = gr.Gallery(
            label="Gallery Row",
            show_label=True,
            columns=15,  # Show 5 images in a row
            object_fit="contain",  # Prevent cropping
            height="100px",  # Set smaller height,
            interactive=False,
            allow_preview=False,
            elem_classes="gray-row",
            scale=9,
        )

    with gr.Row():
        with gr.Column(scale=6):
            im_LDC = gr.ImageEditor(
                value=blank_image(),
                type="numpy",
                crop_size="1:1",
                layers=False,
                height=700,
                brush=gr.Brush(default_size=5),
            )
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("For Annotation"):
                    im_LDC_preview = gr.Image(
                        label="Preview",
                    )

                with gr.TabItem("Previous Annotation"):
                    previous_im_LDC_anntated_preview = gr.Image(label="Preview")

                with gr.TabItem("Edge dispalying"):
                    annotated_edge_preview = gr.Image(label="Preview")

    gr.HTML("<hr style='border:1px solid #ccc;' />")
    with gr.Row():
        download_btn = gr.Button(
            "Download the annotated folder",
            variant="primary",
            elem_classes="tall-button",
            scale=2,
        )
        download_file = gr.File(label="Download File", scale=8)

    # ========== ACTION ==========
    # Download file event
    download_btn.click(
        fn=utils.download_zip_file_LDC,
        inputs=[current_PID],
        outputs=download_file,
    )

    im_LDC.change(
        img_change_action,
        inputs=[im_LDC, current_PID, current_file_name],
        outputs=[
            im_LDC_preview,
        ],
        show_progress="hidden",
    )

    Upload_btn.click(
        load_next_image,
        inputs=[index_state, current_PID, zip_input],
        outputs=[
            im_LDC,
            index_state,
            current_file_name,
            # annotated_kp,
            gallery,
        ],
    )
    gallery.select(
        fn=show_selected_image,
        inputs=[im_LDC, current_PID],  # inpus image to reset the canvas
        outputs=[
            im_LDC,
            current_file_name,
            previous_im_LDC_anntated_preview,
            annotated_edge_preview,
        ],
    )

    demo.load(utils.generate_PID, inputs=None, outputs=current_PID)
if __name__ == "__main__":
    demo.launch(debug=True)
