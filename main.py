from Modules import FaceCameraController

# default : dev / only dir, build
controller = FaceCameraController.FaceCameraController("rpf511", 1, "dev")
controller.run_dev()
# camera1 = FaceCamera.FaceCamera(0, "only dir")
# camera1.run()self.EnvPath = os.path.abspath(os.path.join(os.getcwd(),r'FTRA_core',r'Data',r'Config.txt'))