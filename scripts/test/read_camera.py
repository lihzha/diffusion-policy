# import cv2

# # Set the webcam index (usually starts from 0, but we are using 6 here)
# webcam_index = 4

# # Create a VideoCapture object to access the webcam
# cap = cv2.VideoCapture(webcam_index)

# # Check if the webcam opened correctly
# if not cap.isOpened():
#     print("Error: Could not access webcam.")
#     exit()

# # Get the frame width and height of the video capture
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object to save the video
# output_filename = "output_video.mp4"
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
# out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

# # Start capturing and saving video frames
# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Write the frame to the video file
#     out.write(frame)

#     # Display the frame
#     cv2.imshow("Webcam Feed", frame)

#     # Exit if the user presses the 'q' key
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the video capture and writer objects and close all OpenCV windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()


import os

import cv2

# Define the path to the video file
video_path = "output_video.mp4"

# Create a VideoCapture object to open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video's FPS (Frames per second)
video_fps = cap.get(cv2.CAP_PROP_FPS)

# Set your desired FPS for playback (e.g., 30 FPS)
desired_fps = 5
frame_delay = int(1000 / desired_fps)  # Delay in milliseconds for each frame

# Create a directory to save images if it doesn't exist
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# List to store the last 10 frames
frame_buffer = []

# Frame counter
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Display the current frame
    cv2.imshow("Video Playback", frame)

    # Add the current frame to the buffer
    frame_buffer.append(frame)
    frame_count += 1

    # If buffer exceeds 11 (the current frame + 10 previous frames), pop the oldest frame
    if len(frame_buffer) > 11:
        frame_buffer.pop(0)

    if frame_count > 1000:
        # Wait for keypress and check if it's 'q' to save frames and quit
        key = cv2.waitKey(frame_delay) & 0xFF
    else:
        key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # Save the current frame and the previous 10 frames as images
        for i, img in enumerate(frame_buffer):
            img_filename = f"captured_images/frame_{frame_count - 11 + i}.png"
            cv2.imwrite(img_filename, img)
            print(f"Saved: {img_filename}")

        # break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
