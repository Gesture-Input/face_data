import cv2


while(True):
    key = cv2.waitKey(5) & 0xFF
    if(key != 255):
        print(key)
