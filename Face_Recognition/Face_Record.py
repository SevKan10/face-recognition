import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import multiprocessing as mtp
import os
import Calc_Motor as CM
import math
import time
# Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# FaceNet
model = InceptionResnetV1(pretrained='vggface2').eval()

# Hàm tiền xử lý ảnh
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

# Hàm tính toán khoảng cách giữa hai embedding
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Đọc và tính toán embedding cho ảnh gốc
def compute_reference_embedding(path):
    img = Image.open(path)
    preprocessed_face = preprocess_image(img)
    with torch.no_grad():
        reference_embedding = model(preprocessed_face).detach().numpy().flatten()
    return reference_embedding

# Đọc ảnh gốc và tính toán embedding
reference_image_path = "Face_Recorgnition\Pic_Convert\\"
list = os.listdir(reference_image_path)
img = []
name = []
for i in list:
    reference_embedding = compute_reference_embedding(f"{reference_image_path}{i}")
    img.append(reference_embedding)
    name.append(os.path.splitext(i)[0])

# Hàm so sánh khuôn mặt
state_attract = -1  # Biến để theo dõi đối tượng nhất định
def Scan_and_Record_Face(sample_embedding):
    global state_attract
    Name = "Unknown"
    for i, emb in enumerate(img):
        similarity = cosine_similarity(emb, sample_embedding)
        if similarity > 0.7:
            Name = name[i]
            state_attract = i
            break  # Thêm break để thoát vòng lặp ngay khi tìm thấy khuôn mặt phù hợp
    return Name
Time = 0
state_dir = 0
T = 0
def Attract(sample_embedding, x_pixel, y_pixel, w, w1):
    global state_attract
    global T
    global state_dir
    global Time
    
    similarity = cosine_similarity(img[state_attract], sample_embedding)
    print(similarity)
    if similarity > 0.65:
        
        if x_pixel > w//2 + 20:
            state_dir = 1
            ag = map_value(Calc_Line(x_pixel, y_pixel, w//2, y_pixel), 0, 1000, 254, 0)
            CM.Send(f"R,{int(ag)},{int(ag)},")
        elif x_pixel < w//2 - 20:
            state_dir = 2
            ag = map_value(Calc_Line(x_pixel, y_pixel, w//2, y_pixel), 0, 1000, 254, 0)
            CM.Send(f"L,{int(ag)},{int(ag)},")
        
        # Nếu hình ở trung tâm mà chưa đủ khoảng cách
        if w//2 - 20 <= x_pixel <= w//2 + 10 and w1 < 130: 
            ag = map_value(w1, 0, 1200, 0, 254)
            CM.Send(f"S,{int(ag)},{int(ag)},")
        
        # Nếu hình ở trung tâm và đủ khoảng cách
        if w//2 - 20 <= x_pixel <= w//2 + 10 and w1 >= 130: 
            state_dir = 0
            state_attract = -1
            CM.Send("T")
            

        Time = time.time()
    
    if similarity < 0.65:
        Lost_Object(4)

# Hàm mất đối tượng
def Lost_Object(Wait):
    global Time
    global T
    global state_attract
    global state_dir
    T = time.time()
    if T - Time >= Wait:
        if state_dir == 1:
            CM.Send(f"R,{int(240)},{int(240)},")
        if state_dir == 2:
            CM.Send(f"L,{int(240)},{int(240)},")
            
    if T - Time >= Wait + 2:
        state_dir = 0
        state_attract = -1
        CM.Send("T")
        Time = time.time()
 
# Hàm tính độ dài đoạn thẳng
def Calc_Line(x1, y1, x2, y2):
    result = ((x2 - x1)**2) + ((y2 - y1)**2)
    result = math.sqrt(result)
    return result    

# Hàm Map giá trị
def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Khởi tạo video capture

def Stream(frame_queue, processed_queue):
    last_processed_frame = None
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if ret:
            if frame_queue.empty():
                frame_queue.put(frame.copy())
            
            if not processed_queue.empty():
                last_processed_frame = processed_queue.get()

            if last_processed_frame is not None:
                cv2.imshow("Processed Frame", last_processed_frame)
            else:
                cv2.imshow("Processed Frame", frame)

        if cv2.waitKey(1) == ord('s'):
            break
    
    vid.release()
    frame_queue.put(None)  # Gửi tín hiệu kết thúc cho tiến trình xử lý

def Process(frame_queue, processed_queue):
    CM.begin_COM()
    global state_attract
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break  # Dừng tiến trình khi nhận được tín hiệu kết thúc
            
            h, w, _ = frame.shape  # Lấy chiều cao và chiều rộng của khung hình
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                for detection in results.detections:
                    nose_loc = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                    x = nose_loc.x
                    y = nose_loc.y

                    x_pixel = int(x * w)
                    y_pixel = int(y * h)

                    range = int(Calc_Line(x_pixel, y_pixel, w//2, y_pixel))
                    
                    bboxC = detection.location_data.relative_bounding_box
                    x1, y1, w1, h1 = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    
                    face_img = frame[y1:y1 + h1, x1:x1 + w1]
                    face_pil = Image.fromarray(face_img)
                    preprocessed_face = preprocess_image(face_pil)

                    with torch.no_grad():
                        sample_embedding = model(preprocessed_face).detach().numpy().flatten()
                        if state_attract <= -1:
                            Name = Scan_and_Record_Face(sample_embedding)
                        else: 
                            Attract(sample_embedding, x_pixel, y_pixel, w, w1)

                    frame = cv2.line(frame, (w//2, y_pixel), (x_pixel, y_pixel), (0, 0, 255), 1)
                    frame = cv2.putText(frame, f"Range: {range}", (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(frame, f"Retangle range:{w1}", (10,70),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),  )
                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                    cv2.putText(frame, f'{Name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                     
                processed_queue.put(frame)
            else:
                if state_attract >= 0:
                    print("Lost Object")
                    Lost_Object(4)
                processed_queue.put(frame)

    processed_queue.put(None)  # Gửi tín hiệu kết thúc cho tiến trình hiển thị

if __name__ == "__main__":
    frame_queue = mtp.Queue(maxsize=5)  # Giới hạn kích thước của hàng đợi để tránh tiêu thụ quá nhiều bộ nhớ
    processed_queue = mtp.Queue(maxsize=5)
    
    # Khởi tạo các tiến trình
    capture_process = mtp.Process(target=Stream, args=(frame_queue, processed_queue))
    process_process = mtp.Process(target=Process, args=(frame_queue, processed_queue))

    # Bắt đầu các tiến trình
    capture_process.start()
    process_process.start()

    # Đợi các tiến trình kết thúc
    capture_process.join()
    process_process.join()

    cv2.destroyAllWindows()
