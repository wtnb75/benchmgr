FROM golang:1-alpine AS build
RUN mkdir /src
COPY . /src
RUN cd /src && go build -o empty-go main.go

FROM scratch
COPY --from=build /src/empty-go /
