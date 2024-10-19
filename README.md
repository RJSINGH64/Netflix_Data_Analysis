# Unified Mentor - Netflix Data Visualization and Prediction

Welcome to the **Unified Mentor** project! This project focuses on Netflix data visualization and prediction to determine whether a title is a **movie** or a **TV show**.

**You can watch Project Demo for better understanding**

## Project Overview

In this project, we explore Netflix's dataset, perform data visualization, and build a predictive model to classify titles.

## Project Structure

- **Dashboard App**: The main application interface is built using Streamlit and is located in `dashboard.py`.
  
- **Exploratory Data Analysis (EDA)**: All EDA is done in `research.ipynb`.

- **Model Training**: The model is trained and stored in the `model_training.py`.

## Data Storage

Data is dumped into **MongoDB** using a `.env` file to manage credentials.

# Streamlit app Docker Image

## 1. Login with your AWS console and launch an EC2 instance
## 2. Run the following commands

Note: Do the port mapping to this port:- 8501

```bash
sudo apt-get update -y

sudo apt-get upgrade

#Install Docker

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

```bash
git clone https://github.com/RJSINGH64/Netflix_Data_Analysis.git
```

```bash
cd Netflix_Data_Analysis
```

```bash
docker build -t netflix-app:latest . 
```

```bash
docker images -a  
```

```bash
docker run -d -p 8501:8501 netflix-app
```

```bash
docker ps  
```

```bash
docker stop container_id
```

```bash
docker rm $(docker ps -a -q)
```

```bash
docker login 
```

```bash
docker tag netflix-app:latest  your-docker_username/netflix-app:latest
```

```bash
docker push netflix-app:latest 
```

```bash
docker rmi netflix-app:latest
```

```bash
docker pull netflix-app
```

