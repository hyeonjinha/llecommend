# Ubuntu 기반 이미지 사용
FROM ubuntu:22.04

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ollama 설치
RUN curl -fsSL https://ollama.com/install.sh | sh

# 작업 디렉토리 설정
WORKDIR /app

# EEVE 모델 다운로드
RUN wget https://huggingface.co/teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf/resolve/main/EEVE-Korean-Instruct-10.8B-v1.0-Q5_K_M.gguf

# Modelfile 생성
RUN echo "FROM /app/EEVE-Korean-Instruct-10.8B-v1.0-Q5_K_M.gguf" > Modelfile

# Python 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY ./src/main.py .
COPY ./src/persona.py .
COPY ./src/recommendation.py .
COPY ./src/utils.py .


# Ollama 서비스 시작 및 모델 생성을 위한 스크립트
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
ollama create eeve:latest -f /app/Modelfile\n\
uvicorn main:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

# 포트 8000 노출
EXPOSE 8000

# 컨테이너 시작 시 실행할 명령
CMD ["/app/start.sh"]