package main

import (
	"flag"
	"net"
	"net/http"
	_ "net/http/pprof"
	"strings"
)

func handler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(200)
}
func do_listen(listen string) (net.Listener, error) {
	protos := strings.SplitN(listen, ":", 2)
	switch protos[0] {
	case "unix", "tcp", "tcp4", "tcp6":
		return net.Listen(protos[0], protos[1])
	}
	return net.Listen("tcp", listen)
}

func main() {
	var listen = flag.String("listen", ":3000", "listen address")
	flag.Parse()
	http.HandleFunc("/", handler)
	listener, err := do_listen(*listen)
	if err != nil {
		panic(err)
	}
	http.Serve(listener, nil)
}
