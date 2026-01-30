#! /bin/sh

export -n http_proxy
export -n HTTP_PROXY

[ -f default.pgo ] && mv default.pgo default.pgo.bak
rm -f first second

go build -o first main.go
./first &
fpid=$!
sleep 5

wrk -d 80 http://localhost:3000/ &
wpid=$!
sleep 5
curl -s -o default.pgo 'http://localhost:3000/debug/pprof/profile?seconds=60'
wait $wpid
kill $fpid

go build -o second main.go
./second &
spid=$!
sleep 5
wrk -d 80 http://localhost:3000/
kill $spid

wait
