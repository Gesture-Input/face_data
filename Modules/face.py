import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


import numpy as np


class camera:
    def __init__(self, index, mode = "dev"):
        self.cap = cv2.VideoCapture(index)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)     
        self.success, self.image = self.cap.read()
        self.results = self.face_mesh.process(self.image)
        self.loop = 0
        self.dir_vector = np.array([0.0,0.0,0.0])
        self.dir_state = 0
        self.mode = mode
        
    def camera_update(self):
        self.success, self.image = self.cap.read()
    
    def get_face_mesh_data(self):
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.image)
    
    def draw_face_mesh_data(self):
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=self.image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=self.image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=self.image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe FaceMesh', cv2.flip(self.image, 1))
    
    def get_data(self):
        for face_landmarks in self.results.multi_face_landmarks:
            result = face_landmarks.landmark
        return result
    
    def release(self):
        self.cap.release()
    
    def calculate_face_dir_vector(self,data):
        # 143 : right eye end , 272 : left eye end , 199 : jaw end
        if(self.mode == "dev"):
            print("143 : ",[data[143].x,data[143].y,data[143].z]," 272: ",[data[272].x,data[272].y,data[272].z]," 199 : ",[data[199].x,data[199].y,data[199].z])
        vector_a = np.array([data[199].x,data[199].y,data[199].z]) - np.array([data[143].x,data[143].y,data[143].z])                          
        vector_b = np.array([data[272].x,data[272].y,data[272].z]) - np.array([data[199].x,data[199].y,data[199].z])
        if(self.mode == "dev"):
            print("vector a : ",vector_a," vector b : ",vector_b)
        result = np.cross(vector_a,vector_b) 
        if(self.mode == "dev"):
            print("origin result vector : ",result)
        return result / np.linalg.norm(result)
    
    def current_face_dir_state(self):
#         7,8,9
#         4,5,6       no face : -1
#         1,2,3
        if(self.dir_vector[0] == 0 and self.dir_vector[1] == 0 and self.dir_vector[2] == 0):
            self.dir_state = -1
            return
        self.dir_state = 5
        if(self.dir_vector[1] < -0.2):
            self.dir_state += 3
        elif(self.dir_vector[1] > 0.2):
            self.dir_state -= 3
        
        if(self.dir_vector[0] < -0.4):
            self.dir_state += 1
        elif(self.dir_vector[0] > 0.2):
            self.dir_state -= 1
    
    def current_face_dir_to_text(self):
        res = ""
        if(self.dir_state == -1):
            return "no face"
        
        if(self.dir_state == 5):
            return "front"
        
        if(self.dir_state % 3 == 0):
            res += "right "
        elif(self.dir_state % 3 == 1):
            res += "left "
            
        if(self.dir_state > 6):
            res += "up "
        elif(self.dir_state < 4):
            res += "down "
        return res
        
        
        
    def run(self):
        while self.cap.isOpened():
            self.camera_update()
            if not self.success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            self.get_face_mesh_data()
            
            self.draw_face_mesh_data()
            
            
            if(self.results.multi_face_landmarks == None):
                self.dir_vector = np.array([0,0,0])
                if(self.mode != "build"):
                    print("no face")
                continue
                
            
            print(self.loop)
            data = self.get_data()
            self.dir_vector = self.calculate_face_dir_vector(data)
            self.current_face_dir_state()

            if(self.mode != "build"):
                print("direction : ",self.dir_state," ",self.current_face_dir_to_text())
                print(self.dir_vector)
                
            if(self.mode == "dev"):
                print(self.dir_vector)
                
            self.loop+=1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        self.release()