# học khuông mặt
import cv2
import face_recognition as fr
import os

path = "Python\Face_Recorgnition\Pic\\"
Img_List = os.listdir(path)

index_pic = 0
Name = "Khang"
img = cv2.imread(f"{path}{Img_List[index_pic]}")

# Lấy ảnh gán vào Face recognition
img_black = fr.load_image_file(f"{path}{Img_List[index_pic]}")
img_black = cv2.cvtColor(img_black, cv2.COLOR_BGR2RGB)

# Tìm vị trí khuông mặt
ImgLoc = fr.face_locations(img_black)[0]

# thu gọn bức ảnh chỉ còn khuông mặt
copy = img_black[ImgLoc[0]:ImgLoc[2], ImgLoc[3]:ImgLoc[1]]

cv2.imwrite(f"Python\Face_Recorgnition\Pic_Convert\{Name}.jpg", copy)