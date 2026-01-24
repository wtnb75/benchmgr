FROM python:3 AS wrkunix
RUN git clone https://github.com/wg/wrk
RUN git config --global user.email wtnb75@gmail.com
RUN git config --global user.name "Watanabe Takashi"
WORKDIR wrk
RUN git remote add unix https://github.com/kazeburo/wrk && git fetch unix
RUN git cherry-pick 4051f747bdc08b3bac685c29b0fadafd19e33097
RUN make

FROM python:3
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | bash
RUN apt update && apt install -y wrk apache2-utils hey task nghttp2-client --no-install-recommends
ADD requirements.txt /
RUN pip install -r /requirements.txt
COPY --from=wrkunix /wrk/wrk /usr/local/bin/wrk-unix
