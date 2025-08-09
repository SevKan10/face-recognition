import serial

# Khởi tạo kết nối Arduino
def begin_COM():
    global arduino
    arduino = serial.Serial(port='COM11', baudrate=115200, timeout=.1)

def Send(Data):
    arduino.write((Data + '\n').encode())

def reset():
    global speed
    speed = 0
    global speed_up 
    speed_up = 0
