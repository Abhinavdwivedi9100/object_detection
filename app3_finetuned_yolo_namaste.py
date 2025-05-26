import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
# model = YOLO(r'C:\Users\HP\OneDrive - softsensor.ai\pvr\weights\best.pt')
# model = YOLO(r'C:\Users\HP\OneDrive - softsensor.ai\pvr\weights\dhariya.pt')
# model = YOLO(r'C:\Users\HP\OneDrive - softsensor.ai\pvr\weights\model_120624.pt')
# model = YOLO(r'C:\Users\HP\OneDrive - softsensor.ai\pvr\weights\model_140624.pt')
# model = YOLO(r'C:\Users\HP\OneDrive - softsensor.ai\pvr\weights\greet_yawn_helmet_smoke_mask.pt')
model = YOLO(r'new_model_18072024.pt')

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pil_image = Image.fromarray(img)

        results = model.predict([pil_image], conf=0.28)
        annotated_image = None

        if len(results) > 0:  
            r = results[0]  
            im_bgr = r.plot()  
            annotated_image = im_bgr[..., ::-1] 

        return annotated_image if annotated_image is not None else img

def main():
    st.title("Live-Cam Surveillance with Softsensor AI")


    webrtc_streamer(key="example", video_transformer_factory=YOLOVideoTransformer)

if __name__ == "__main__":
    main()