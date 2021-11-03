FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy

ADD . /app

RUN pipenv install -r requirements.txt

COPY ["xgb_predict.py","xgb_model.bin","./"]

EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:9696", "xgb_predict:app" ]