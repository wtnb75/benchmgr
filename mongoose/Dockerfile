FROM alpine:3 AS build
RUN apk add build-base
ADD 7.20.tar.gz /
COPY main.c /mongoose-7.20/
WORKDIR /mongoose-7.20
RUN gcc -O9 -o /server main.c -I. mongoose.c

FROM alpine:3
COPY --from=build /server /
EXPOSE 8000/tcp
