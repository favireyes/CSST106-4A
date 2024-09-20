# Machine Problem 2: Applying Image Processing Techniques
## Image Transformations
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display two images side by side
def display_image(img, title="Image"):
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.axis('off')
  plt.show()
```
```
from google.colab import files
from io import BytesIO
from PIL import Image

# Upload an image
uploaded1 = files.upload()

# Convert to OpenCV format
image_path1 = next(iter(uploaded1)) # Get the image file name
image1 = Image.open(BytesIO(uploaded1[image_path1]))
image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
display_image(image1, "Original Image")
```
![original(reyes)](https://github.com/user-attachments/assets/bb40a71e-dd59-4c9e-8e4e-481cbfb7f5ce)

### Scaling and Rotation
```
# Scaling
def scale_image(img, scale_factor):
  height, width = img.shape[:2]
  scaled_img = cv2.resize(img, (int(width * scale_factor), int(height *scale_factor)), interpolation=cv2.INTER_LINEAR)
  return scaled_img

# Rotate
def rotate_image(img, angle):
  height, width = img.shape[:2]
  center = (width // 2, height // 2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_img = cv2.warpAffine(img, matrix, (width, height))
  return rotated_img

# Scale image by 0.5
scaled_image1 = scale_image(image1, 0.5)
display_image(scaled_image1, "Scaled Image (50%)")

# Rotate image by 45 degrees
rotated_image1 = rotate_image(image1, 45)
display_image(rotated_image1, "Rotated Image (45%)")
```
![scaled(reyes)](https://github.com/user-attachments/assets/d884a68e-7246-4bf1-b47e-fedb465d1de1)
![rotated(reyes)](https://github.com/user-attachments/assets/777083fd-1323-4900-b454-f1583fb1e09c)

### Blurring and Edge Detection
```
# Gaussian Blur
gaussian_blur1 = cv2.GaussianBlur(image1, (5, 5), 0)
display_image(gaussian_blur1, "Gaussian Blur (5x5)")

# Median Blur
median_blur1 = cv2.medianBlur(image1, 5)
display_image(median_blur1, "Median Blur (5x5)")
```
![gaussian(reyes)](https://github.com/user-attachments/assets/f5bd739c-224b-455f-a97c-7bd01c360c45)
![median(reyes)](https://github.com/user-attachments/assets/0235c6a5-4552-4ee0-b886-499f6b252495)

```
# Canny Edge Detection
edges1 = cv2.Canny(image1, 100, 200)
display_image(edges1, "Canny Edge Detection (100, 200)")
```

## Problem-Solving Session
Scenario:

It is needed to process an image of a road from a drone feed for vehicle detection. The road is at an angle, the image is noisy, and the vehicles need to be identified.

Solution:

To process an image of a road for vehicle detection, the first step will be to rotate the image using affine transformation to ensure the road is properly aligned. This will allow for better analysis of objects within the scene. Next, Gaussian blurring will be applied to reduce noise in the image, helping to smooth out imperfections without losing important details. Once noise is minimized, the image will be downscaled to reduce its size and optimize it for quicker processing. Finally, edge detection techniques, such as Canny or Sobel edge detection, will be applied to identify vehicles on the road based on their edges, enhancing the precision of vehicle detection.

### Upload Image
```
from google.colab import files
from io import BytesIO
from PIL import Image

# Upload an image
uploaded2 = files.upload()

# Convert to OpenCV format
image_path2 = next(iter(uploaded2)) # Get the image file name
image2 = Image.open(BytesIO(uploaded2[image_path2]))
image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
display_image(image2, "Original Image")
```
![original(drone)](https://github.com/user-attachments/assets/5b5f1ec6-ee6f-41a7-b2cb-3993939fc481)

### Rotate
```
# Rotate
def rotate_image(img, angle):
  height, width = img.shape[:2]
  center = (width // 2, height // 2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_img = cv2.warpAffine(img, matrix, (width, height))
  return rotated_img

# Rotate image by 90 degrees
rotated_image2 = rotate_image(image2, 90)
display_image(rotated_image2, "Rotated Image (90°)")
```
![rotated(drone)](https://github.com/user-attachments/assets/2b49a84b-1105-489f-bbbf-5a7ad63f5ea9)

### Applying Gaussian BLur
```
# Apply Gaussian Blur
def apply_gaussian_blur(img, kernel_size=(5, 5), sigma=0):
    # kernel_size should be odd numbers like (3, 3), (5, 5), etc.
    blurred_img = cv2.GaussianBlur(img, kernel_size, sigma)
    return blurred_img

# Apply Gaussian Blur to the rotated image
blurred_image2 = apply_gaussian_blur(rotated_image2, (5, 5), 0)

# Display the blurred image
display_image(blurred_image2, "Blurred Image")
```
![blurred(drone)](https://github.com/user-attachments/assets/b37831dd-e9e4-4785-b659-6100b3274fda)

### Scaling Blurred Image
```
# Scale
def scale_image(img, scale_factor):
    height, width = img.shape[:2]
    scaled_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
    return scaled_img

# Scale the blurred image
scaled_blurred_image2 = scale_image(blurred_image2, 0.75)

# Display the scaled blurred image
display_image(scaled_blurred_image2, "Scaled Blurred Image (75%)")
```
![scaled(drone)](https://github.com/user-attachments/assets/2440a20c-7b1b-432d-81eb-e35fdce57d69)

### Edge Detection using Canny
```
# Apply Canny Edge Detection
def apply_canny_edge_detection(img, low_threshold, high_threshold):
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges

# Set thresholds for Canny Edge Detection
low_threshold = 100
high_threshold = 200

# Apply Canny Edge Detection to the scaled blurred image
edges_image2 = apply_canny_edge_detection(scaled_blurred_image2, low_threshold, high_threshold)

# Display the edges image
display_image(edges_image2, "Canny Edge Detection on Scaled Image")
```
![canny(drone)](https://github.com/user-attachments/assets/a05381ae-24f9-4c38-97ae-49c2e832dfc6)

### Edge Detection using Sobel
```
# Apply Sobel Edge Detection
def apply_sobel_edge_detection(img):
    # Calculate the gradients in the x and y direction
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction

    # Combine the gradients
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Convert to uint8
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    return sobel_combined

# Apply Sobel Edge Detection to the scaled blurred image
sobel_image2 = apply_sobel_edge_detection(scaled_blurred_image2)

# Display the Sobel edges image
display_image(sobel_image2, "Sobel Edge Detection on Scaled Image")
```
![sobel(drone)](https://github.com/user-attachments/assets/50c2614f-39a1-4038-b527-6c4058a4900f)

```
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(rotated_image2, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (90°)")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(blurred_image2, cv2.COLOR_BGR2RGB))
plt.title("Blurred Image")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(scaled_blurred_image2, cv2.COLOR_BGR2RGB))
plt.title("Scaled Blurred Image (75%)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(edges_image2, cmap='gray')
plt.title("Canny Edge Detection on Scaled Image")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(sobel_image2, cmap='gray')
plt.title("Sobel Edge Detection on Scaled Image")
plt.axis('off')

plt.tight_layout()
plt.show()
```
![combined](https://github.com/user-attachments/assets/b45ca0fd-0bcb-496f-aef3-e30e5f940753)

## Assignment: Implementing Image Transformations and Filtering

In this activity, I applied various image processing techniques, including scaling, rotation, blurring, and edge detection, to two sample images: one clear and one noisy/low-quality. The goal was to compare the results of these techniques on both types of images, observing how noise and image quality impact the outcomes of different transformations. I documented each step and analyzed the effectiveness of each technique.

### Define Image Processing Functions

```
def display_images(img1, img2, title1="Image 1", title2="Image 2"):
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis('off')
    
    plt.show()

# Scaling
def scale_image(img, scale_factor):
    height, width = img.shape[:2]
    scaled_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
    return scaled_img

# Rotation
def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, matrix, (width, height))
    return rotated_img

# Blurring (Gaussian)
def apply_gaussian_blur(img, kernel_size=(5, 5), sigma=0):
    blurred_img = cv2.GaussianBlur(img, kernel_size, sigma)
    return blurred_img

# Canny Edge Detection
def apply_canny_edge_detection(img, low_threshold, high_threshold):
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges

# Sobel Edge Detection
def apply_sobel_edge_detection(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined
```

### Load the Images

```
# image3 (clear) and image4 (noisy)
image3 = cv2.imread('clear_image.jpg') 
image4 = cv2.imread('noisy_image.jpg')

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
plt.title("Image 3 (Clear)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
plt.title("Image 4 (Noisy)")
plt.axis('off')

plt.tight_layout()
plt.show()
```
![original(c n)](https://github.com/user-attachments/assets/8dbb1b72-8180-41fc-ae16-4cd7365f3530)

* Clear Image: A high-quality, clear image was selected for comparison purposes.
* Noisy Image: A noisy, low-quality version of the clear image was chosen to demonstrate how noise affects the results of image processing techniques.

### Apply Scaling to Clear and Noisy Image

Both images were resized using bilinear interpolation. Images were scaled down to 50% of their original size.

```
scaled_image3 = scale_image(image3, 0.5)
scaled_image4 = scale_image(image4, 0.5)
display_images(scaled_image3, scaled_image4, "Scaled Clear Image", "Scaled Noisy Image")
```
![scaled(c n)](https://github.com/user-attachments/assets/29f84698-27a6-4ec2-b975-baa178e619e2)

The clear image retained more details after scaling, whereas the noisy image showed amplified distortions as the pixels became more compressed.

### Rotate Clear and Noisy Image

Affine transformation was applied to rotate both images by 45°.

```
rotated_image3 = rotate_image(image3, 45)
rotated_image4 = rotate_image(image4, 45)
display_images(rotated_image3, rotated_image4, "Rotated Clear Image", "Rotated Noisy Image")
```
![rotated(c n)](https://github.com/user-attachments/assets/e8e66157-1281-42c8-92e1-8cf6630bf805)

The clear image maintained its shape and visual appeal. However, the noisy image experienced minimal distortion around the edges, and the noise artifacts became more pronounced due to rotation.

### Apply Gaussian blur to Clear and Noise Image

Gaussian blur was applied to reduce noise, particularly in the noisy image.

```
blurred_image3 = apply_gaussian_blur(image3, (5, 5), 0)
blurred_image4 = apply_gaussian_blur(image4, (5, 5), 0)
display_images(blurred_image3, blurred_image4, "Blurred Clear Image", "Blurred Noisy Image")
```
![blurred(c n)](https://github.com/user-attachments/assets/d13ecf2d-b600-4420-87b5-3bfda11b3ef2)

Blurring worked effectively to reduce noise in the noisy image, though some fine details were lost. In the clear image, blurring reduced sharpness but did not significantly alter the overall quality. 

### Apply Canny and Sobel Edge Detection to Clear and Noisy Image

Canny edge detection was used with threshold values of 100 and 200 to identify edges in both images. Sobel edge detection was applied, computing gradients in both x and y directions.

```
edges_image3 = apply_canny_edge_detection(image3, 100, 200)
edges_image4 = apply_canny_edge_detection(image4, 100, 200)
display_images(edges_image3, edges_image4, "Canny Edges (Clear)", "Canny Edges (Noisy)")
```
![canny(c n)](https://github.com/user-attachments/assets/019a2124-d383-45bd-ba1d-3e1b6b966b2c)

In the clear image, edges were cleanly detected with minimal noise. However, in the noisy image, edge detection produced many false positives due to noise, making it difficult to differentiate meaningful edges from noise artifacts.

```
sobel_image3 = apply_sobel_edge_detection(image3)
sobel_image4 = apply_sobel_edge_detection(image4)
display_images(sobel_image3, sobel_image4, "Sobel Edges (Clear)", "Sobel Edges (Noisy)")
```
![sobel(c n)](https://github.com/user-attachments/assets/6e747d6e-2d01-4b1e-b3a9-f3e91831ffef)

Sobel edge detection highlighted the clear image’s edges well, but in the noisy image, the presence of noise led to additional gradients, producing less distinct and noisier edges.

### Challenges

An issue that was encountered in the implementation of image processing techniques was the enhancement of noise distortions in the noisy image during scaling and rotation, which made the succeeding edge identification less accurate. Both Canny and Sobel edge detection algorithms struggled to effectively detect edges in the noisy image, as the embedded noise caused numerous false positives and a significant drop in edge sharpness.

### Conclusion

On the clear image, the procedures of blurring, rotating, scaling, and edge detection worked well with little loss in quality. On the other hand, because the noise had a substantial impact on the outcomes, the noisy image needed to be handled with more caution, especially during edge identification. Preprocessing methods like noise reduction or denoising filters can improve the results of the noisy image and help with later transformations, particularly edge recognition.

