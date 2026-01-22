use std::{convert::Infallible, io, time::Duration, env};

use actix_http::{HttpService, Request, Response, StatusCode};
use actix_server::Server;

#[actix_rt::main]
async fn main() -> io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    Server::build()
        .bind("test", (env::args().nth(1).unwrap(), env::args().nth(2).unwrap().parse().unwrap()), || {
            HttpService::build()
                .client_request_timeout(Duration::from_secs(1))
                .finish(|_: Request| async move {
                    Response::build(StatusCode::OK);
                    Ok::<_, Infallible>("")
                })
                .tcp()
        })?
        .workers(4)
        .run()
        .await
}
