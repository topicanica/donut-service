FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload model
RUN python -c "from transformers import DonutProcessor, VisionEncoderDecoderModel; DonutProcessor.from_pretrained('hyunguk1/donut-base-receipt-v2'); VisionEncoderDecoderModel.from_pretrained('hyunguk1/donut-base-receipt-v2')"

COPY app/ ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
