import os
import subprocess

VENV_NAME = "env_" + os.path.basename(os.path.dirname(os.path.abspath(__file__)))

subprocess.run(["python3", "-m", "venv", VENV_NAME])
subprocess.run([f"{VENV_NAME}/bin/python3", "-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([f"{VENV_NAME}/bin/python3", "-m", "pip", "install", "-r", "requirements.txt"])
