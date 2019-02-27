FROM alpine
RUN apk add --no-cache curl
RUN mkdir /model
RUN curl -O http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz \
    && tar -xvzf araneum_none_fasttextcbow_300_5_2018.tgz -C /model \
    && rm araneum_none_fasttextcbow_300_5_2018.tgz \
    && ls /model