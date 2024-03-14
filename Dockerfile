FROM python:3.10-bookworm
COPY lip_translate lip_translate
COPY requirements.txt requirements.txt
# RUN pip install -U pip cython wheel
RUN pip install -r requirements.txt
CMD uvicorn lip_translate.api:app --host 0.0.0.0
