FROM rhangelxs/russian_embeddings:serve AS serve

FROM python:3.6
ENV TZ UTC

RUN apt-get update && apt-get install --no-install-recommends -y \
  curl && \
  apt-get clean autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/{apt,dpkg,cache,log}/

ADD ./requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt

ADD . /opt/models/araneum_none_fasttextcbow_300_5_2018
RUN touch /opt/models/__init__.py
RUN touch /opt/__init__.py

WORKDIR /opt/models/araneum_none_fasttextcbow_300_5_2018

RUN curl -O http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz \
    && tar -xvzf araneum_none_fasttextcbow_300_5_2018.tgz -C . \
    && rm araneum_none_fasttextcbow_300_5_2018.tgz \
    && ls .

COPY --from=serve /opt/serve /opt/serve
RUN pip install --no-cache-dir -r /opt/serve/requirements.txt

ENV FLASK_APP serve/wsgi.py
ENV FLASK_CONFIG Dev
ENV FLASK_DEBUG 1

EXPOSE 8080

WORKDIR /opt

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8080"]