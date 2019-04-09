FROM rhangelxs/russian_embeddings:serve AS serve

FROM python:3.6
ENV TZ UTC

COPY --from=serve /opt/serve /opt/serve
WORKDIR /opt/serve
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install --no-install-recommends -y \
  curl && \
  apt-get clean autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/{apt,dpkg,cache,log}/

ADD ./requirements.txt /opt/requirements.txt

WORKDIR /opt
#RUN mkdir /model
RUN curl -O http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz \
    && tar -xvzf araneum_none_fasttextcbow_300_5_2018.tgz -C . \
    && rm araneum_none_fasttextcbow_300_5_2018.tgz \
    && ls .

ADD . /opt/models/araneum_none_fasttextcbow_300_5_2018
RUN touch /opt/models/__init__.py
RUN touch /opt/__init__.py

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080

ENV FLASK_APP serve/wsgi.py
ENV FLASK_CONFIG Dev
ENV FLASK_DEBUG 1

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8080"]