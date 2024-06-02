# CapstoneProject-2

This code is designed to detect faces in a video, predict whether the faces are real or fake using a deep learning model, and annotate the video with the prediction results. The script utilizes several libraries, including OpenCV, Dlib, and PyTorch, to achieve this functionality. Below is a detailed explanation of the code.

Key Components
Libraries and Imports:

os and argparse: For handling file paths and command-line arguments.
cv2 (OpenCV): For video processing.
dlib: For face detection.
torch: For loading and running the deep learning model.
PIL.Image and tqdm: For image handling and progress display.
Functions:

get_boundingbox: Calculates a bounding box around a detected face with optional scaling and minimum size.
preprocess_image: Converts an image to a tensor suitable for input to the deep learning model.
predict_with_model: Uses the deep learning model to predict whether a face is real or fake.
test_full_image_network: Main function that processes the video frame by frame, detects faces, predicts their authenticity, and writes the annotated frames to an output video.
Detailed Breakdown
get_boundingbox(face, width, height, scale=1.3, minsize=None)
Purpose: Calculate the bounding box for a detected face.
Inputs:
face: The detected face object from Dlib.
width, height: Dimensions of the image.
scale: Scale factor to adjust the bounding box size.
minsize: Minimum size for the bounding box.
Outputs: Coordinates and size of the bounding box.
preprocess_image(image, cuda=True)
Purpose: Prepare an image for input into the deep learning model.
Inputs:
image: The image to preprocess.
cuda: Boolean to decide whether to use GPU.
Outputs: Preprocessed image tensor.
predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True)
Purpose: Predict whether a face is real or fake using the model.
Inputs:
image: The image of the face.
model: The pre-trained deep learning model.
post_function: The function to apply to model output.
cuda: Boolean to decide whether to use GPU.
Outputs: Prediction label and output tensor.
test_full_image_network(video_path, model_path, output_path, start_frame=0, end_frame=None, cuda=True)
Purpose: Process a video, detect faces, make predictions, and save annotated video.
Inputs:
video_path: Path to the input video.
model_path: Path to the pre-trained model.
output_path: Directory to save the output video.
start_frame, end_frame: Frame range to process.
cuda: Boolean to decide whether to use GPU.
Process:
Load video and initialize face detector and model.
Process each frame to detect faces and predict their authenticity.
Annotate frames and write to output video.
Main Execution Block
Uses argparse to handle command-line arguments.
Calls test_full_image_network with provided arguments to process the video(s).
Example Usage
Single Video:

sh
Copy code
python script.py --video_path path/to/video.mp4 --model_path path/to/model.pth --output_path path/to/output
Directory of Videos:

sh
Copy code
python script.py --video_path path/to/videos --model_path path/to/model.pth --output_path path/to/output
Sample Output
Original Frame:

Annotated Frame:

In the annotated frame, detected faces will be surrounded by bounding boxes, and the label "real" or "fake" will be displayed based on the prediction.
