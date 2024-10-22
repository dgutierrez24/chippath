# chippath
Uncover your future in the Semiconductor Industry
-------

## Mission Statement: 
To foster economic opportunity and drive the microelectronics

industry’s growth, we create pathways and opportunities for job seekers and provide

tools and systems for semiconductor companies to attract, develop, retain, and advance

a diverse and skilled workforce. and skilled workforce.

-------

Built with `Python 3.10.15`.

### 1. On-Premise Install:
Create Environment and Install Requirements
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Run
```
streamlit run app.py
```
-------
### 2. Docker Image Install:

Create your Dockerfile:

```
# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Go to directory and run:
```
docker build -t my_streamlit_app .
```
Once image is built start Docker Container:
```
docker run -p 8501:8501 my_streamlit_app
```
**Troubleshoot Notes:** Change ports as needed

-------


### 3. AWS EC2 Instance Install:
Computing requirements:
- RAM 8GBs
- CPU cores 4
- OS - Ubuntu

#### Instructions:
Create an EC2 instance which has the above mentioned computing power. You need to edit inbound rules of the 
instance to run the streamlit on a global port so that you can access the server from anywhere.
After this, all the instructions are same as local setup. 

**Troubleshoot Notes:**
You may encounter error while creating a virtual environment. Run `apt install python3.10-venv`