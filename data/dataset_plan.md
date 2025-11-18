# Dataset Plan

## 1. Dataset Sources
- Real CCTV footage (if available)
- Open-source datasets: COCO, OpenImages, CrowdHuman
- Custom collected images/videos

## 2. Required Classes
- Person
- Suspicious object (bag, weapon)
- Unusual posture (running, falling)

## 3. Folder Structure
data/
│── raw/
│── processed/
│── videos/
│── images/
│── test/

## 4. Annotation Format
Using YOLO format:
class x_center y_center width height

