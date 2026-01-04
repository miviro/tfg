import os
import json
import cv2
from tqdm import tqdm

def process_video(video_path, json_path, output_images_dir, output_labels_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.basename(os.path.dirname(video_path))

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_name}"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx_str = str(frame_idx).zfill(6)
        
        if frame_idx < len(data['exist']) and data['exist'][frame_idx] == 1:
            image_name = f"{video_name}_{frame_idx_str}"
            image_path = os.path.join(output_images_dir, f"{image_name}.jpg")
            cv2.imwrite(image_path, frame)
            
            label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
            
            with open(label_path, 'w') as label_file:
                rect = data['gt_rect'][frame_idx]
                x, y, w, h = rect
                
                # Convert to YOLO format
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height
                
                label_file.write(f"0 {x_center} {y_center} {norm_w} {norm_h}\n")
                
    cap.release()

def main():
    base_dir = 'datasets/Anti-UAV-RGBT'
    output_base_dir = 'datasets/Anti-UAV-RGBT-YOLO'

    for split in ['train', 'test', 'val']:
        input_dir = os.path.join(base_dir, split)
        output_images_dir = os.path.join(output_base_dir, split, 'images')
        output_labels_dir = os.path.join(output_base_dir, split, 'labels')
        
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        if not os.path.exists(input_dir):
            print(f"Skipping {split} as directory {input_dir} does not exist.")
            continue

        video_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        for video_dir in video_dirs:
            video_path = os.path.join(input_dir, video_dir, 'visible.mp4')
            json_path = os.path.join(input_dir, video_dir, 'visible.json')

            if os.path.exists(video_path) and os.path.exists(json_path):
                process_video(video_path, json_path, output_images_dir, output_labels_dir)
            else:
                print(f"Skipping {os.path.join(input_dir, video_dir)}: missing video or json file.")

if __name__ == '__main__':
    main()
