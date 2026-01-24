package main

import (
	"flag"
	"net/http"
	_ "net/http/pprof"
)

func handler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(200)
}

func main() {
	var listen = flag.String("listen", ":3000", "listen address")
	flag.Parse()
	http.HandleFunc("/", handler)
	http.ListenAndServe(*listen, nil)
}
