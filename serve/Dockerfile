FROM python:3.6
ENV TZ UTC
ADD ./requirements.txt /opt/serve/requirements.txt
WORKDIR /opt/serve

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENV FLASK_APP wsgi.py
ENV FLASK_CONFIG Dev
ENV FLASK_DEBUG 1
ADD . /opt/serve

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8080"]
