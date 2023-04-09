from Modules import FaceCamera
from Modules import DataService
import numpy as np
import os
import platform
import cv2


class FaceCameraController:
    def __init__(self,name,camera_index,dev_set):
        self.camera = FaceCamera.FaceCamera(camera_index,dev_set)
        self.data_service = DataService.DataService(name)
        self.params = self.data_service.get_params()
        if len(self.params) != 0:
            self.camera.set_params(self.params)
        self.current_camera_loc = np.array([0,0,0])
        self.key_input = -1
        # 0 : main page | 1 : init page
        self.console_page = 0
        self.console_page_status = 0
        

    def handle_input(self, key):
        self.key_input = key & 0xFF
        if self.key_input == 27:
            return -1

        if(self.console_page == 0):
            if(self.key_input == ord('0')):
                self.console_page = 1
                return 0
        
        if(self.console_page == 1):
            if(self.key_input == ord('0')):
                self.console_page_status = 1
                return 1
        return 0

    def handle_status(self):
        if(self.console_page == 1):
            #camera distance init
            if(self.console_page_status == 1):
                return
        return


    def print_state(self):
        print(self.camera.current_face_dir_to_text())
        print("dir vector "+str(self.camera.dir_vector)+" ")
        print("eye  "+self.camera.current_eye_to_text() +
              str(self.camera.eye))
        print("face loc polar   "+str(self.camera.face_loc))
        if(self.console_page == 0):
            print("0 : initialize camera | ")
            return
        if(self.console_page == 1):
            if(self.console_page_status == 0):
                print("Place your face 1m away from the center of the camera. press 0")

    def run_dev(self):
        while self.camera.cap.isOpened():
            self.camera.camera_update()
            if not self.camera.success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            self.camera.get_face_mesh_data()
            
            self.camera.draw_face_mesh_data()

            if(self.camera.OS == "Windows"):
                os.system('cls')
            else:
                os.system('clear')
            # print(self.camera.results.detection)
            # if (self.camera.results.multi_face_landmarks == None):
            #     # self.camera.dir_vector = np.array([0,0,0])
            #     if (self.camera.mode != "build"):
            #         print("no face")
            #     continue
            if self.camera.results.multi_face_landmarks:
                
                print(self.camera.loop)
                data = self.camera.get_data()
                self.camera.dir_vector = self.camera.calculate_face_dir_vector(data)
                self.camera.face_loc = self.camera.get_loc_polar(self.camera.get_face_loc(data))
                self.camera.eye = self.camera.check_eye(data)
                self.camera.current_face_dir_state()

            self.print_state()

            self.camera.loop += 1
            if(self.handle_input(cv2.waitKey(5)) == -1):
                break
            
        
        self.camera.release()