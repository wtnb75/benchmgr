#include <arpa/inet.h>
#include <dlfcn.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#ifdef DEBUG
#define debug_print printf
#else
#define debug_print(...) ;
#endif

char* _ischange(const struct sockaddr* addr, socklen_t addrlen) {
  char host[NI_MAXHOST];
  char service[NI_MAXSERV];
  int result =
      getnameinfo(addr, addrlen, host, sizeof(host), service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV);
  if (result == 0) {
    debug_print("addr: %s:%s\n", host, service);
  } else {
    debug_print("getnameinfo failed: %s\n", gai_strerror(result));
  }
  char* port = getenv("SOCKUNIX_PORT");
  char* nextpath = getenv("SOCKUNIX_PATH");
  if (port == NULL || nextpath == NULL) {
    debug_print("target/nextpath is null: %s %s\n", port, nextpath);
    return NULL;
  }
  if (strcmp(service, port) == 0) {
    debug_print("hit: %s -> %s\n", service, nextpath);
    return nextpath;  // do not free
  }
  debug_print("miss: %s target=%s\n", port, service);
  return NULL;
}

int _orig_bind(int sockfd, const struct sockaddr* addr, socklen_t addrlen) {
  static int (*origfn)(int sockfd, const struct sockaddr* addr, socklen_t addrlen) = NULL;
  if (!origfn) {
    origfn = (int (*)(int, const struct sockaddr*, socklen_t))dlsym(RTLD_NEXT, "bind");
    debug_print("get next func bind: %p\n", origfn);
  }
  return origfn(sockfd, addr, addrlen);
}

int _orig_connect(int sockfd, const struct sockaddr* addr, socklen_t addrlen) {
  static int (*origfn)(int sockfd, const struct sockaddr* addr, socklen_t addrlen) = NULL;
  if (!origfn) {
    origfn = (int (*)(int, const struct sockaddr*, socklen_t))dlsym(RTLD_NEXT, "connect");
    debug_print("get next func connect: %p\n", origfn);
  }
  return origfn(sockfd, addr, addrlen);
}

int bind(int sockfd, const struct sockaddr* addr, socklen_t addrlen) {
  debug_print("preload function(bind)\n");
  char* path = _ischange(addr, addrlen);
  if (path) {
    unlink(path);
    int newfd = socket(AF_UNIX, SOCK_STREAM, 0);
    int sockfd2 = dup2(newfd, sockfd);
    debug_print("newfd = %d, dup = %d\n", newfd, sockfd2);
    if (newfd != sockfd2) {
      close(newfd);
    }
    struct sockaddr_un next_addr;
    next_addr.sun_family = AF_UNIX;
    strncpy(next_addr.sun_path, path, sizeof(next_addr.sun_path) - 1);
    return _orig_bind(sockfd2, (const struct sockaddr*)&next_addr, sizeof(next_addr));
  }
  return _orig_bind(sockfd, addr, addrlen);
}

int connect(int sockfd, const struct sockaddr* addr, socklen_t addrlen) {
  debug_print("preload function(connect)\n");
  char* path = _ischange(addr, addrlen);
  if (path) {
    int newfd = socket(AF_UNIX, SOCK_STREAM, 0);
    int sockfd2 = dup2(newfd, sockfd);
    debug_print("newfd = %d, dup = %d\n", newfd, sockfd2);
    if (newfd != sockfd2) {
      close(newfd);
    }
    struct sockaddr_un next_addr;
    next_addr.sun_family = AF_UNIX;
    strncpy(next_addr.sun_path, path, sizeof(next_addr.sun_path) - 1);
    return _orig_connect(sockfd2, (const struct sockaddr*)&next_addr, sizeof(next_addr));
  }
  return _orig_connect(sockfd, addr, addrlen);
}
