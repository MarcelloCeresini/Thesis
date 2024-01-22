# Thesis

## Setup the virtual environment with base dependencies (Linux)

Create the environment, activate it and upgrade pip:

```bash
python -m venv env_Thesis
source env_Thesis/bin/activate
python -m pip install --upgrade pip
```

and run

```bash
pip install -r requirements.txt 
```

Then we need to clone inside the environment a forked repository, through terminal:

```bash
cd env_Thesis/lib/python3.11/site-packages
git clone https://github.com/MarcelloCeresini/meshio.git
```

or through GitHub Desktop, in the following folder:

```bash
cd env_Thesis/lib/python3.11/site-packages
```

## Setup for Windows

Make sure to have the capability to run scripts with:

```powershell
Set-ExecutionPolicy unrestricted
```

Create the venv, activate it and install requirements:

```powershell
python -m venv env_Thesis
.\envThesis\Scripts\activate
python -m pip install --upgrade pip
```

Clone a forked repository inside the environment:

```powershell
cd .\env_Thesis\Lib\site-packages\
git clone https://github.com/MarcelloCeresini/meshio.git
```

or though GitHub Desktop, in the following folder:

```powershell
cd .\env_Thesis\Lib\site-packages\
```
