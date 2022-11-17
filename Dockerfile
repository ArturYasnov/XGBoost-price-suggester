# Placeholder Dockerfile, adapt to your needs!

#RUN adduser -D -H adevinta
#USER adevinta

FROM python:3.8.1

COPY src /ml
# COPY models /ml/models
# COPY resources /ml/resources

ADD requirements.txt /ml/requirements.txt

WORKDIR /ml

RUN pip install -r /ml/requirements.txt

EXPOSE 8080
CMD python inference.py
