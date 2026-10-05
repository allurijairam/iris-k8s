FROM python:3.12-slim

workdir /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

copy app/ .
RUN test -f /app/model/MLmodel || (echo " Model Directory is missing" && exit 1)

RUN useradd -u 1000 appuser && chown -R appuser /app

USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]