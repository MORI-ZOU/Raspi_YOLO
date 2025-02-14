import streamlit as st
import cv2
import numpy as np
from picamera2 import Picamera2  # Picamera2 ライブラリをインポート
from ultralytics import YOLO
import tempfile
import datetime
import os


def main():
    st.title("Pi Camera YOLO Test")

    st.sidebar.title("Setting")
    with st.sidebar.form("special form"):
        uploaded_file = st.file_uploader("Load weights file", type=["pt"])
        iou_threshold = st.slider("IOU", 0.0, 1.0, 0.5, 0.01)
        conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.5, 0.01)
        is_submit = st.form_submit_button("Submit")

    if "model" not in st.session_state:
        st.session_state.model = None

    if uploaded_file is not None and is_submit:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(uploaded_file.read())
            weight_path = tmp_file.name
        st.session_state.model = YOLO(weight_path)
        st.write(f"アップロードされたモデルを読み込みました: {datetime.datetime.now()}")
    elif is_submit:
        st.session_state.model = YOLO("yolov8n.pt")
        st.write(f"プリトレインモデルを読み込みました: {datetime.datetime.now()}")

    # Picamera2 を使って画像を取得する
    try:
        picam2 = Picamera2()
        config = picam2.create_still_configuration()  # 静止画用の設定
        picam2.configure(config)
        picam2.start()
        cv2_img = picam2.capture_array()  # NumPy 配列として画像を取得（BGR形式）
        picam2.close()  # 使用後はカメラを閉じる
    except Exception as e:
        st.error(f"カメラのキャプチャに失敗しました: {e}")
        return

    if cv2_img is not None:
        if not st.session_state.model:
            st.sidebar.warning("ウェイトファイルをアップロードしてください")
            st.image(cv2_img, channels="BGR")
            return

        # YOLO による検出
        results = st.session_state.model.predict(source=cv2_img, device="cpu", conf=conf_threshold, iou=iou_threshold, save=False)
        detected_img = results[0].plot(conf=False, labels=False)
        st.image(detected_img, channels="BGR", caption="Detected Image")

        # 画像の保存
        save_dir = f"data/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        dt_now = datetime.datetime.now().strftime("%H%M%S")
        output_path = os.path.join(save_dir, f"{dt_now}.jpg")
        cv2.imwrite(output_path, cv2_img)

        # 検出結果をテキストとして保存
        for res in results:
            boxes = res.boxes
            classes = res.boxes.cls
            xywh = boxes.xywhn
            with open(os.path.join(save_dir, f"{dt_now}.txt"), "w") as f:
                for i, box in enumerate(xywh):
                    class_id = classes[i]
                    x, y, w, h = box
                    f.write(f"{int(class_id)} {x} {y} {w} {h}\n")

        st.write(f"検出画像を保存しました：{output_path}")
        with open(output_path, "rb") as f:
            st.download_button(label="Download detected image", data=f.read(), file_name="detected_image.jpg", mime="image/jpeg")


if __name__ == "__main__":
    main()
