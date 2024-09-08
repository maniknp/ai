```bash
!git clone https://github.com/facebookresearch/segment-anything-2.git
%cd segment-anything-2
!pip install -e .
```

```bash
!pip install segment-anything opencv-python numpy pillow face_recognition dlib
```

```bash
%cd /content/segment-anything-2
%cd checkpoints
!./download_ckpts.sh
%cd ..
```


```bash
import torch
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import face_recognition
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Function to download image from URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Download placeholder images (replace these URLs with your actual image files)
source_image = download_image("https://filmfare.wwmindia.com/content/2020/jul/shahrukhkhan41596116759.jpg")
target_image = download_image("https://cdn.siasat.com/wp-content/uploads/2024/01/Sallu.jpg")
family_photo = download_image("https://feeds.abplive.com/onecms/images/uploaded-images/2023/04/18/19a3085bf722da7c75f9708da30bc8d71681789348745431_original.jpg?impolicy=abp_cdn&imwidth=1200&height=675")

# Convert images to RGB
source_image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
family_photo_rgb = cv2.cvtColor(family_photo, cv2.COLOR_BGR2RGB)

# Display the input images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(source_image_rgb)
plt.title("Your Image (Source)")
plt.subplot(1, 3, 2)
plt.imshow(target_image_rgb)
plt.title("Friend's Image (Target)")
plt.subplot(1, 3, 3)
plt.imshow(family_photo_rgb)
plt.title("Family Photo")
plt.show()

# Load the SAM2 model
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Detect faces in the family photo
family_face_locations = face_recognition.face_locations(family_photo_rgb)
family_face_encodings = face_recognition.face_encodings(family_photo_rgb, family_face_locations)

# Get the encoding of your face from the source image
your_face_encoding = face_recognition.face_encodings(source_image_rgb)[0]

# Find your face in the family photo
your_face_location = None
for i, face_encoding in enumerate(family_face_encodings):
    match = face_recognition.compare_faces([your_face_encoding], face_encoding)[0]
    if match:
        your_face_location = family_face_locations[i]
        break

if your_face_location is None:
    print("Your face was not found in the family photo.")
else:
    print("Your face was found in the family photo.")

    # Use SAM2 to create a precise mask for your face in the family photo
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(family_photo_rgb)
        top, right, bottom, left = your_face_location
        center_point = np.array([(left + right) // 2, (top + bottom) // 2])
        input_point = np.array([center_point])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label)

    mask = masks[0]

    # Prepare the target (friend's) face for swapping
    target_face_location = face_recognition.face_locations(target_image_rgb)[0]
    target_landmarks = face_recognition.face_landmarks(target_image_rgb)[0]
    your_landmarks = face_recognition.face_landmarks(family_photo_rgb, [your_face_location])[0]

    # Calculate transformation matrix
    source_points = np.array([your_landmarks['left_eye'][0], your_landmarks['right_eye'][0], your_landmarks['nose_tip'][0]], dtype=np.float32)
    target_points = np.array([target_landmarks['left_eye'][0], target_landmarks['right_eye'][0], target_landmarks['nose_tip'][0]], dtype=np.float32)
    M = cv2.getAffineTransform(target_points, source_points)

    # Apply transformation to align target face with your face in the family photo
    aligned_target_face = cv2.warpAffine(target_image_rgb, M, (family_photo_rgb.shape[1], family_photo_rgb.shape[0]))

    # Perform face swapping
    result = family_photo_rgb.copy()
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    result = np.where(mask_3d, aligned_target_face, result)

    # Smooth the edges of the swapped face
    kernel = np.ones((5,5), np.uint8)
    mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    mask_blurred = cv2.GaussianBlur(mask_dilated.astype(np.float32), (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    mask_blurred = (mask_blurred * 255).astype(np.uint8)

    # Apply the blurred mask for smoother blending
    mask_blurred_3d = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2) / 255.0
    result = (result * mask_blurred_3d + family_photo_rgb * (1 - mask_blurred_3d)).astype(np.uint8)

    # Display the result
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(family_photo_rgb)
    plt.title("Original Family Photo")
    plt.subplot(1, 3, 2)
    plt.imshow(target_image_rgb)
    plt.title("Friend's Face (Swapped In)")
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title("Face Swapped Result")
    plt.show()

```


