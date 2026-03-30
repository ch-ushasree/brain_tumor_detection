=> Brain tumors are among the most aggressive and life-threatening forms of cancer, necessitating prompt and accurate diagnosis for effective treatment. Traditional methods of detecting brain tumors, such as manual analysis of Magnetic Resonance Imaging (MRI) scans, can be time-consuming and prone to human error. With the advancement of deep learning technologies, automated methods for brain tumor detection have gained significant attention. This project aims to leverage Convolutional Neural Networks (CNNs) to develop a robust and efficient model for brain tumor detection from MRI images, implemented in Python.

=> The aim of this project is to develop an automated and efficient method for detecting brain tumors in MRI images using a Convolutional Neural Network (CNN) architecture. By leveraging the capabilities of deep learning, the project seeks to improve the accuracy, speed, and reliability of brain tumor detection, thereby assisting medical professionals in early diagnosis and treatment planning. The ultimate goal is to create a robust tool that can aid in the timely identification of brain tumors, potentially leading to better patient outcomes and more effective clinical interventions.

=> The model leverages the powerful image processing capabilities of CNNs to automatically learn and extract relevant features from the input images, minimizing the need for manual feature extraction. The dataset used for training and validation comprises a diverse set of brain MRI images, pre-processed to enhance image quality and ensure consistency. The CNN architecture is designed with multiple convolutional layers, activation functions, pooling layers, and fully connected layers to achieve high accuracy and robustness.

=> Convolutional Neural Networks (CNNs):
   CNNs are a class of deep learning models that have demonstrated exceptional performance in various image processing tasks. They are particularly well-suited for tasks involving spatial hierarchies, such as image classification, object detection, and medical image analysis. CNNs automatically learn to extract relevant features from images through multiple layers of convolutional operations, pooling layers, and activation functions


=> Model Architecture
The CNN model designed for this project consists of three convolutional layers, each followed by a ReLU activation function to introduce non-linearity. The final layer employs a sigmoid activation function to output a segmentation mask, indicating the presence or absence of a tumor in each pixel of the MRI image. The architecture is as follows:
1.	Conv1: Convolutional layer with 16 filters, kernel size of 3x3, and padding of 1.
2.	Conv2: Convolutional layer with 32 filters, kernel size of 3x3, and padding of 1.
3.	Conv3: Convolutional layer with 1 filter, kernel size of 3x3, and padding of 1.
4.	ReLU: Activation function applied after each convolutional layer.
5.	Sigmoid: Activation function applied at the final layer to produce a probability map for tumor presence.


=> The dataset for this project comprises MRI images of the brain, annotated with tumor regions. Each image undergoes preprocessing steps to enhance quality and ensure consistency, including:
1.	Grayscale Conversion: MRI images are converted to grayscale if they are in color.
2.	Resizing: Images are resized to a fixed dimension (e.g., 256x256 pixels) to standardize input to the CNN.
3.	Normalization: Pixel values are normalized to a range of 0 to 1 to facilitate faster and more stable training.

=> What It Does:
1. Upload an MRI image and BrainScan will:
2. Classify whether the scan shows signs of a tumor (Yes / No)
3. Segment the tumor region and overlay it on the original scan
4. Display the original image, prediction mask, and overlay side by side
5. Show a confidence score so you know how certain the model is

**This isn't meant to replace a radiologist — it's a decision-support tool designed to assist with preliminary screening.**

=> Demo : Run the app locally at http://localhost:8000 after setup 

=> Project structure
braintumorscan/
├── models/        # CNN & U-Net models
├── utils/         # Preprocessing & visualization
├── weights/       # Trained model files (.pth)
├── app.py         # Main app
├── server.py      # Runs local server
├── requirements.txt

=> Requirements:
Software Requirements
•	Operating System: Compatible with major operating systems (Windows, Linux, macOS).
•	Python: Python 3.7 or higher.
 Install Dependencies
bashpip install -r requirements.txt
Development Environment
•	IDE/Text Editor: An Integrated Development Environment (IDE) or text editor for writing and debugging the code (e.g., PyCharm, Visual Studio Code, Jupyter Notebook).
Data Preprocessing Tools
•	Image Processing Software: Tools like OpenCV or other image processing libraries for additional preprocessing tasks if needed.

=> How to Use
Running the App
bashpython server.py
Then open your browser and go to:
http://localhost:8000
Launch the app using the command above
Upload a brain MRI scan (grayscale or RGB, .jpg/.png/.bmp)
Click Analyze — the model runs inference on your image


View Results:
Left panel: original MRI
Middle panel: predicted segmentation mask
Right panel: overlay showing where the tumor was detected
The confidence score appears below the result panels
