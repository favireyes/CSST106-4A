# Machine Problem 1: Exploring the Role of Computer Vision and Image Processing in AI

https://github.com/user-attachments/assets/50cad7e8-88dc-4a37-bcf1-d13563668dcb

## Introduction to Computer Vision and Image Processing
**Computer Vision** is a branch of AI that uses sensing devices, algorithms, and deep learning to interpret and understand visual data, similar to human vision. It automates tasks like recognizing objects and detecting patterns. 

Its applications include content organization, text extraction, augmented reality, autonomous vehicles, manufacturing, spatial analysis, and face recognition. Computer vision's versatility and importance in various industries, including healthcare, highlight its widespread use. 

Modern computer vision relies on deep learning and neural networks for improved analysis.

## Types of Image Processing Techniques
### Object Detection
Using algorithms to identify and localize specific objects within an image and create bounding boxes around them.

Example: Models can accurately detect multiple objects in real-time, including people, vehicles, and animals, in surveillance videos or autonomous vehicles.

### Image Cropping
Selectively extracts a specific part of an image to focus on an area of interest, removing unwanted parts while preserving the desired area.

Example: Cropping a face from a group photo to use as a profile picture, or isolating a product in an image for e-commerce listings.

### Image Manipulation
Modifying an image to achieve specific visual effects, involving techniques like combining images, adding text, and modifying image attributes.

Example: Using software to create collages, design marketing materials, or edit photos to adjust colors and add annotations.

## Case Study Overview
This project utilizes computer vision to detect faces and create personalized ID cards. A DNN model is used to identify faces within an image, altering the image format to identify them and drawing boxes around them. This method is faster and more accurate than manually cropping faces. The cropped faces are then fitted onto an ID card template, along with other user information. Automating these steps makes creating ID cards easier and ensures consistent creation at a higher rate.

This method addresses issues of manual processing, accuracy, and efficiency, making it suitable for rapid identification visuals.

## Image Processing Implementation
For the problem of face detection and personalized ID card creation, a model that combines a deep neural networK for face detection with image manipulation techniques for generating ID cards is used.

### Face Detection
The model starts with an image containing faces, converts it into a blob for the pre-trained DNN to detect them, and then draws bounding boxes around them. The DNN then provides the coordinates of these boxes, indicating the location of faces in the image.

```def detect_face_dnn(image_path):
    # Load the DNN model
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return None

    (h, w) = image.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform inference and get the faces
    detections = net.forward()

    # Find the face with the highest confidence
    max_confidence = 0
    face = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Ensure the confidence is above a threshold
        if confidence > 0.5 and confidence > max_confidence:
            max_confidence = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            padding = 100 
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(w, endX + padding)
            endY = min(h, endY + padding)

            # Extract the face
            face = image[startY:endY, startX:endX]

    return face
```

### Image Cropping
The model uses bounding box coordinates from the face detection step to crop out areas containing detected faces, resulting in isolated face images for further processing.

```box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
(startX, startY, endX, endY) = box.astype("int")

padding = 100  
startX = max(0, startX - padding)
startY = max(0, startY - padding)
endX = min(w, endX + padding)
endY = min(h, endY + padding)

# Extract the face
face = image[startY:endY, startX:endX]
pe("int")
face = image[startY:endY, startX:endX]
```

### ID Card Creation
The model uses cropped face images and a predefined ID card template to create a personalized ID card with user-specific information, such as the user's name, using image manipulation techniques.

```def create_id_card(face_image, name, save_path):
    id_template = Image.open("ID Card.png")

    # Convert the face image from OpenCV format to PIL format
    image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

    # Resize the image using LANCZOS resampling (formerly known as ANTIALIAS)
    face_pil = image.resize((290, 290), Image.Resampling.LANCZOS)

    #(290, 290) is the size of the image panel
    #starting point: (155, 250) end point: (435, 530) | w = 435 - 155 h = 530 - 250

    # Paste the face onto the ID template
    id_template.paste(face_pil, (150, 245, 440, 535))

    # (150, 245, 440, 535) is the starting point and endpoint of the image panel of the template

    # Import the font
    font = ImageFont.truetype('glacial-indifference.bold.otf', size=23)
    draw = ImageDraw.Draw(id_template)

    # Calculate the bounding box of the text
    text_bbox = draw.textbbox((0, 0), name, font=font)

    # Calculate the width of the text
    text_width = text_bbox[2] - text_bbox[0]

    # Calculate the x-coordinate to center the text
    x_position = (id_template.width - text_width) // 2

    # Draw the text centered
    draw.text((x_position, 706), name, font=font, fill='#04294f')

    # x_position is the calculated center of the bounding box, 706 is the y_position

    # Display the finished ID card
    id_template.show()

    # Save the finished ID card
    id_template.save(save_path)
```

The model automates face detection and ID card creation, reducing manual effort and accelerating workflow. It uses a DNN for high accuracy, maintains consistency, and operates efficiently, making it ideal for large-volume ID card production.

![image](https://github.com/user-attachments/assets/e003a0e7-a513-4858-878e-3dd2529120ed)
![image (1)](https://github.com/user-attachments/assets/7fa6003f-5d0a-4e76-92b0-e681b5c38eb5)
![image (2)](https://github.com/user-attachments/assets/8900601c-099e-4181-8eb9-cebd54b3fa93)
![image (3)](https://github.com/user-attachments/assets/1071d8f7-5550-4235-8fdf-d0aba3ddc233)

## Conclusion
AI uses effective image processing to understand and interpret data in a visual format with a great deal of accuracy. An enhancement technique, object recognition, and face detection amongst others are techniques that offer an aid to various applications, which range from security and surveillance applications to automatically working photo editing of medical imagery.

This activity made me understand that image processing develops the quality and usability of an image while automatic processes, on the other hand, would be very time-consuming and prone to mistakes. This showed that advanced image processing techniques can greatly improve how efficiently, accurately, and consistently AI applications work.

## Extension Activity
### Self-Supervised Learning
**Self-supervised learning (SSL)** is transforming fields such as computer vision and natural language processing by eliminating the need for considerable manual data labeling. Traditional supervised learning relies on large labeled datasets, which can be time-consuming and expensive to develop.

SSL enables models to learn from unlabeled data by creating tasks that automatically infer important information. This method greatly reduces the time and expense associated with data labeling, making advanced AI systems more accessible and inexpensive. SSL improves model performance and generalization across multiple tasks by learning robust features from unlabeled input. It also accelerates the training process, especially in fields where data labeling is difficult or expensive. SSL approaches are extensively applicable, increasing the efficiency and capability of AI models in both natural language processing and computer vision. 

Overall, SSL is a significant improvement in AI, promising to improve performance, lower costs, and speed up the creation of future AI systems.

## References
What is Computer Vision? | Microsoft Azure. (n.d.). https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-computer-vision#object-classification

Simplilearn. (2024, August 13). What is image processing : overview, applications, benefits, and more. Simplilearn.com. https://www.simplilearn.com/image-processing-article#types_of_image_processing

Kundu, R. (2024, July 25). Image Processing: Techniques, Types, & Applications [2024]. V7. https://www.v7labs.com/blog/image-processing-guide#image-processing-techniques

What is Self-Supervised Learning? | IBM. (n.d.). https://www.ibm.com/topics/self-supervised-learning

