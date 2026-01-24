#! /bin/sh
set -eu
version=7.20

[ -f ${version}.tar.gz ] || \
    curl -LO https://github.com/cesanta/mongoose/archive/refs/tags/${version}.tar.gz
