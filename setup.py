import os
import subprocess

VENV_NAME = "env_" + os.path.basename(os.path.dirname(os.path.abspath(__file__)))

subprocess.run(["python3", "-m", "venv", VENV_NAME])
subprocess.run(["source", VENV_NAME + "/bin/activate"])
subprocess.run(["pip", "install", "-r", "requirements.txt"])


