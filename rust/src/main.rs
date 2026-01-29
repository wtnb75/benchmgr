use std::{convert::Infallible, env, io, time::Duration};

use actix_http::{HttpService, Request, Response, StatusCode};
use actix_server::Server;
use std::thread::available_parallelism;

#[actix_rt::main]
async fn main() -> io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    let workers = available_parallelism().unwrap().get();
    let host = env::args().nth(1).unwrap();
    let port = env::args().nth(2).unwrap().parse().unwrap();

    Server::build()
        .bind("test", (host, port), || {
            HttpService::build()
                .client_request_timeout(Duration::from_secs(1))
                .finish(|_: Request| async move {
                    Response::build(StatusCode::OK);
                    Ok::<_, Infallible>("")
                })
                .tcp()
        })?
        .workers(workers)
        .run()
        .await
}
