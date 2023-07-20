FROM ubuntu:20.04

COPY . .

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install -r requirements.txt

WORKDIR /src

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000