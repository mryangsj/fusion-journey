import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# %% Setup torch device and data type
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# %% Setup global variables
size_yolo = 640
size_lcm = 512
size_cv_window = 720
alpha_mask = 0.3  # transparency factor of people mask (0-1)


# %% Load YOLO and LCM
# yolo_path = "yolov8n-seg.pt"
yolo_path = "yolov8n-seg.torchscript"
ctrl_net_path = "models/control_v11p_sd15_canny"
lcm_path = "models/LCM_Dreamshaper_v7"

yolo = YOLO(yolo_path)

ctrl_net = ControlNetModel.from_pretrained(ctrl_net_path, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(lcm_path, controlnet=ctrl_net, safety_checker=None).to(device)
pipe.enable_attention_slicing()

# %% Fully use CPU
torch.set_num_threads(8)  # set the number of threads to use in CPU

# Load camera
# cap = cv2.VideoCapture("passenger.mp4")
cap = cv2.VideoCapture(2)  # 0 for built-in webcam, 1 for iphone camera, 2 for OBS camera

# Get the resolution of the video
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Compute the center crop start and end points
cap_size = min(cap_width, cap_height)
cap_crop_x, cap_crop_y = (cap_width - cap_size) // 2, (cap_height - cap_size) // 2

# Load the background image
img_back = cv2.imread("image/Funes_Bolzano.png")

# Crop the background image to a square
img_back_width, img_back_height = img_back.shape[:2]
img_back_size = min(img_back_width, img_back_height)
img_back_crop_x, img_back_crop_y = (img_back_width - img_back_size) // 2, (img_back_height - img_back_size) // 2
img_back = img_back[img_back_crop_y:img_back_crop_y + img_back_size, img_back_crop_x:img_back_crop_x + img_back_size]

# Resize the background image to the LCM size
img_back = cv2.resize(img_back, (size_lcm, size_lcm))

# Set CV window
name_window = "Results"
cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(name_window, size_cv_window, size_cv_window)


# Get prompt
def get_prompt():
    prompt = "landscape of Dolomiti, goat, sheep, cow, bird, house, sunny, highly detailed, 8k"
    return prompt


# %%
# LCM diffusion parameters
negative_prompt = "bad anatomy, deformed, ugly, disfigured, low quality, bad quality, sketches"
num_inference_steps = 4
guidance_scale = 4.0
controlnet_conditioning_scale = 0.9

while True:
    # ========================
    # YOLO segmentation
    # ========================

    # Read the frame
    ret, frame = cap.read()
    if not ret: break

    # Crop the frame to a square
    frame = frame[cap_crop_y:cap_crop_y + cap_size, cap_crop_x:cap_crop_x + cap_size]

    # Resize the frame for YOLO
    frame = cv2.resize(frame, (size_yolo, size_yolo))

    # Perform YOLO segmentation
    result = yolo(frame, verbose=True)[0]

    if result.masks is not None:
        # Filter the person class
        masks = result.masks.data
        boxes = result.boxes.data
        labels = boxes[:, 5].int()
        person_masks = masks[labels == 0]

        # Resize the frame and person masks to the LCM size
        frame = cv2.resize(frame, (size_lcm, size_lcm))
        person_masks = person_masks.unsqueeze(1)
        person_masks = torch.nn.functional.interpolate(person_masks, (size_lcm, size_lcm), mode="nearest").squeeze(1)

        # Create a person mask
        person_mask = torch.sum(person_masks, dim=0)

        # Denote person with color in the frame
        frame = cv2.resize(result.plot(), (size_lcm, size_lcm))

    # ========================
    # LCM diffusion
    # ========================

        # Mask the background image
        img_comp = torch.from_numpy(img_back).to(device)
        img_comp[person_mask > 0] = 0

        # Canny edge detection
        img_canny = cv2.Canny(img_comp.cpu().numpy(), 100, 200)
        img_canny = np.stack([img_canny] * 3, axis=-1)

        # Generate final image with LCM
        with torch.inference_mode():
            image = pipe(
                prompt                        = get_prompt(),
                negative_prompt               = negative_prompt,
                image                         = Image.fromarray(img_canny),
                width                         = size_lcm,
                height                        = size_lcm,
                num_inference_steps           = num_inference_steps,
                guidance_scale                = guidance_scale,
                controlnet_conditioning_scale = controlnet_conditioning_scale,
                generator                     = torch.Generator(device=device).manual_seed(19),
                guess_mode                    = True,
                output_type                   = "np",
            ).images[0]

        # Convert the generated image to BGR and uint8
        image = (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255.0).astype(np.uint8)

        # image = np.zeros((size_lcm, size_lcm, 3), np.uint8)

    # ========================
    # Display results
    # ========================
        cv2.imshow(name_window, np.vstack((
            np.hstack((frame, img_back)),
            np.hstack((img_canny, image))
        )))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()