FROM rust:1-alpine AS build
RUN mkdir /src
COPY . /src
RUN cd /src && cargo build --release

FROM alpine:3
COPY --from=build /src/target/release/empty-rs /
