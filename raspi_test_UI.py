import streamlit as st
import cv2
import numpy as np
from raspi_camera import USBCamera
from ultralytics import YOLO
import tempfile
import datetime
import os

def main():
    st.title("USB Camera YOLO Test")

    st.sidebar.title("setting")
    with st.sidebar.form("special form"):
        uploaded_file=st.file_uploader("Load weights file", type=["pt"])
        iou_threshold=st.slider("IOU", 0.0, 1.0, 0.5, 0.01)
        conf_threshold=st.slider("Confidence", 0.0, 1.0, 0.5, 0.01)
        is_submit = st.form_submit_button("submit")

    if 'model' not in st.session_state:
        st.session_state.model=None

    if uploaded_file is not None and is_submit:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(uploaded_file.read())
            weight_path=tmp_file.name
        st.session_state.model=YOLO(weight_path)
        st.write(f"アップロードされたモデルを読み込みました:{datetime.datetime.now()}")
    elif is_submit:
        st.session_state.model=YOLO("yolov8n.pt")
        st.write(f"プリトレインモデルを読み込みました:{datetime.datetime.now()}")

    image_file_buffer=st.camera_input("Take a picture")

    if image_file_buffer is not None:
        bytes_date=image_file_buffer.getvalue()

        cv2_img=cv2.imdecode(np.frombuffer(bytes_date, np.uint8), cv2.IMREAD_COLOR)

        if not st.session_state.model:
            st.sidebar.warning("Upload weights file")
            st.image(cv2_img, channels="BGR")
            return

        # model=YOLO(weight_path)

        results=st.session_state.model.predict(source=cv2_img, device='cpu',conf=conf_threshold, iou=iou_threshold, save=False)
        detected_img=results[0].plot(conf=False, labels=False)
        st.image(detected_img, channels="BGR", caption="detected image")

        #save
        save_dir=f"data/{datetime.datetime.now().strftime('')}"
        os.makedirs(save_dir, exist_ok=True)
        dt_now=datetime.datetime.now().strftime('%H%M%S')
        output_path= os.path.join(save_dir,f"{dt_now}.jpg")
        cv2.imwrite(output_path, cv2_img)

        #save txt
        for res in results:
            #boxとクラスIDを取得する
            boxes=res.boxes
            classes=res.boxes.cls

            #xywh形式でバウンディングボックスのデータを取得する
            xywh=boxes.xywhn

            with open(os.path.join(save_dir,f"{dt_now}.txt"), 'w') as f:
                for i, box in enumerate(xywh):
                    #バウンディングボックスの座標とクラスIDを書き込み
                    class_id=classes[i]
                    x,y,w,h=box
                    f.write(f"{int(class_id)} {x} {y} {w} {h}\n")

        st.write(f"検出画像を保存しました：{output_path}")
        st.download_button(label="Download detected image", data=open(output_path, "rb").read(), file_name="detected_image.jpg", mime="image/jpeg")
   

if __name__=="__main__":
    main()