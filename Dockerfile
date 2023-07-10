FROM python:3.10-slim

COPY . .

RUN apt-get update

RUN pip3 install -r requirements.txt

WORKDIR /src

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000