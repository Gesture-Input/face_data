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
            
            if(self.camera.results.multi_face_landmarks == None):
                # self.camera.dir_vector = np.array([0,0,0])
                if(self.camera.mode != "build"):
                    print("no face")
                continue
                
            
            print(self.camera.loop)
            data = self.camera.get_data()
            self.camera.dir_vector = self.camera.calculate_face_dir_vector(data)
            self.camera.face_loc = self.camera.calculate_face_loc(data)
            self.camera.eye = self.camera.check_eye(data)
            self.camera.current_face_dir_state()

            if(self.camera.mode != "build"):
#                 print("direction : ",self.camera.dir_state," ",self.camera.current_face_dir_to_text())
                print(self.camera.current_face_dir_to_text())
                print("dir vector "+str(self.camera.dir_vector)+" ")
                print("eye  "+self.camera.current_eye_to_text()+str(self.camera.eye))
                print("face loc   "+str(self.camera.face_loc))
                

                
                
                
            self.camera.loop+=1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        self.camera.release()