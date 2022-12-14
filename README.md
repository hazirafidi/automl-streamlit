# Automl Webapp with Pycaret

## Method 1: Run locally by cloning this Repository
1. Open your Terminal and run the following command
```
$ git clone https://github.com/hazirafidi/automl.git
```
2. Create virtual environment 
```
$ conda env create -f env.yml
```
3. Activate your virtual environment
```
$ conda activate env
```
4. Run Webapp
```
$ streamlit run src/🏠Home.py
```

## Method 2: Run with Docker
1. Install Docker Desktop
2. Run the followng command in terminal to create docker container
```
$ docker run -p 8501:8501 streamlit-autmol
```
