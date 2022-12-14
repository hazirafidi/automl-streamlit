FROM python:3.8
EXPOSE 8501
WORKDIR /automl
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/hazirafidi/automl.git .
RUN pip3 install -r requirements.txt
CMD ["streamlit", "run", "src/🏠Home.py"]