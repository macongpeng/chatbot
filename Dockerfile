FROM python:slim

RUN apt-get update && apt-get install gcc g++ clang -y

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "./medibotbeflask.py"]