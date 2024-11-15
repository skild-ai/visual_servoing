# test script to check how a model performs for a single image

from ultralytics import YOLOv10
import cv2
import os

# Load the pretrained YOLOv10 model
# model = YOLOv10.from_pretrained('jameslahm/yolov10s')  # 's' can be replaced with n/m/b/l/x as needed
model = YOLOv10("../test/weights_m/spoon_combined.pt")

# Load the image
image_path = '../test/spatula/8.jpg'
# image_path = "/home/unitree/poorvi/data/detection/traj4/torso_cam/rgb/2.png"
image = cv2.imread(image_path)

# Perform prediction
results = model.predict(source=image, conf=0.1)  # Adjust the confidence threshold if needed

# Filter and display only bounding boxes with label "spoon"
for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())  # Get the class ID as an integer

        # Check if the label corresponds to "spoon"
        if model.names[class_id] == "spoon":
            x1, y1, x2, y2 = box.xyxy[0]  # Use [0] to get the first (and only) row
            confidence = box.conf.item()  # Convert tensor to a scalar

            # Draw bounding boxes on the original image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{model.names[class_id]}: {confidence:.2f}', 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

# Save or display the output image
# output_path = "../test/output_new.png"
output_path = image_path.replace("test", "test_output")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, image)