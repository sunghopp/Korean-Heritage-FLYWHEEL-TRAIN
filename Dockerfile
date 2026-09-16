FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT ["python", "-m", "flywheel.pipeline"]
