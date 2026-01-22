FROM python:3
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | bash
RUN apt update && apt install -y wrk apache2-utils hey task --no-install-recommends
ADD requirements.txt /
RUN pip install -r /requirements.txt
