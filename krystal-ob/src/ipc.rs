use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use crate::mqtt::EdgeStatusState;
use std::sync::atomic::Ordering;
use log::{info, error};

#[derive(Serialize, Deserialize, Debug)]
pub struct ObDirective {
    pub pause_mining: bool,
    pub overclock_active: bool,
    pub max_gpu_percent_allowed: i32,
}

pub async fn start_ipc_server(socket_path: &str, edge_state: Arc<EdgeStatusState>) {
    // Odstranenie stareho socketu ak existuje
    let _ = std::fs::remove_file(socket_path);

    let listener = match UnixListener::bind(socket_path) {
        Ok(l) => l,
        Err(e) => {
            error!("Nepodarilo sa vytvorit IPC socket: {}", e);
            return;
        }
    };

    info!("IPC server bezi na {}", socket_path);

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((mut stream, _)) => {
                    let state_ref = edge_state.clone();
                    tokio::spawn(async move {
                        loop {
                            let priority = state_ref.priority.load(Ordering::Relaxed);
                            // Logika: ak priority > 5, pauzujeme mining.
                            let directive = ObDirective {
                                pause_mining: priority > 5,
                                overclock_active: priority <= 5,
                                max_gpu_percent_allowed: if priority > 5 { 0 } else { 100 },
                            };
                            
                            if let Ok(json) = serde_json::to_string(&directive) {
                                let mut packet = json.into_bytes();
                                packet.push(b'\n'); // newline delimiter
                                if let Err(_) = stream.write_all(&packet).await {
                                    break;
                                }
                            }
                            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        }
                    });
                }
                Err(e) => {
                    error!("IPC Accept error: {}", e);
                }
            }
        }
    });
}
