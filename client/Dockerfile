FROM golang:1 AS gowrk
RUN go install github.com/tsliwowicz/go-wrk@latest

FROM rust:1 AS rewrk
RUN cargo install rewrk --git https://github.com/ChillFish8/rewrk.git

FROM python:3
COPY --from=gowrk /go/bin/go-wrk /usr/local/bin/
COPY --from=rewrk /usr/local/cargo/bin/rewrk /usr/local/bin/
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | bash
RUN apt update && apt install -y wrk apache2-utils hey task nghttp2-client --no-install-recommends
ADD requirements.txt /
RUN pip install -r /requirements.txt
