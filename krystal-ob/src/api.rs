use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::Arc;
use crate::ObMonitor;
use log::{info, error};

pub async fn start_rest_api(monitor: Arc<ObMonitor>) {
    let listener = match TcpListener::bind("0.0.0.0:8080").await {
        Ok(l) => l,
        Err(e) => {
            error!("Nepodarilo sa vytvorit REST API server: {}", e);
            return;
        }
    };

    info!("REST API (Glass Brain Data Exporter) bezi na http://0.0.0.0:8080/telemetry");

    tokio::spawn(async move {
        loop {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mon_ref = monitor.clone();
                tokio::spawn(async move {
                    let mut buffer = [0; 1024];
                    if let Ok(bytes_read) = stream.read(&mut buffer).await {
                        if bytes_read > 0 {
                            let json_data = crate::get_telemetry_json(&mon_ref);
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                                json_data
                            );
                            let _ = stream.write_all(response.as_bytes()).await;
                        }
                    }
                });
            }
        }
    });
}
