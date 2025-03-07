import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

def segment_image_smp(model_path, input_image_path, output_mask_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model and load weights
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    img = cv2.imread(input_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_img)  # (1, 1, H, W)
        prob_map = torch.sigmoid(output)
        mask_pred = (prob_map > 0.5).float().cpu().numpy()[0, 0]
    
    mask_pred_uint8 = (mask_pred * 255).astype(np.uint8)
    cv2.imwrite(output_mask_path, mask_pred_uint8)
    print(f"Segmentation mask saved to {output_mask_path}")

# Example usage:
segment_image_smp("smp_unet_placenta.pth", "data/images/test_image.jpg", "test_mask_pred.png")


def draw_circles_on_masked_image(input_image_path, mask_path, output_annotated_path):
    import cv2
    import numpy as np
    
    # Load the original image and the predicted mask
    img = cv2.imread(input_image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw minimum enclosing circles around each contour
    annotated = img.copy()
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(annotated, center, radius, (0, 255, 0), 2)
    
    cv2.imwrite(output_annotated_path, annotated)
    print(f"Annotated image saved to {output_annotated_path}")

# Example usage:
draw_circles_on_masked_image("data/images/test_image.jpg", "test_mask_pred.png", "test_image_annotated.jpg")