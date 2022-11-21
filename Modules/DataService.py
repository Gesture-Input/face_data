import numpy as np
import os
import platform


class DataService:
    def __init__(self,name):
        if(platform.system() == "Windows"):
            self.user_data_path = os.getcwd() + "\\Data\\" + name +".txt" 
        else:
            self.user_data_path = os.getcwd() + "/Data/" + name +".txt" 
        self.params = []            

    def get_params(self):
        if len(self.params) != 0 :
            return self.params
        else:
            if(not os.path.exists(self.user_data_path)):
                print("file not exists")
                return []
            data = open(self.user_data_path,"r", encoding='utf-8') 
            line = data.readline()
            split_data = line[:-1].split(" ")
            result = list(map(float, split_data))
            data.close()
            self.params = result
            return result

    def save_params(self,data):
        if len(data) != 0 :
            self.user_data = data
        file = open(self.user_data_path, "w")
        result = ""
        for temp in self.user_data:
            result += temp + " "
        result += "\n"
        file.write(result)
        file.close()
        return