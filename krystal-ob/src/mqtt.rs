use rumqttc::{AsyncClient, MqttOptions, QoS, Event, Incoming};
use std::sync::atomic::{AtomicI32, AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::Duration;
use log::{info, error};
use serde::Deserialize;

#[derive(Deserialize)]
struct EdgeStatusPayload {
    priority_task: bool,
    reserved_gpu_percent: Option<i32>,
    priority_level: Option<i32>,
}

pub struct EdgeStatusState {
    pub priority: AtomicI32,
    pub task_active: AtomicBool,
}

impl EdgeStatusState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            priority: AtomicI32::new(0),
            task_active: AtomicBool::new(false),
        })
    }
}

pub async fn start_mqtt_listener(state: Arc<EdgeStatusState>) {
    let mut mqttoptions = MqttOptions::new("krystal-ob-governor", "localhost", 1883);
    mqttoptions.set_keep_alive(Duration::from_secs(5));

    let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);
    
    // Subscribe to topic
    if let Err(e) = client.subscribe("krystal/edge/status", QoS::AtMostOnce).await {
        error!("Nepodarilo sa prihlasit na MQTT: {}", e);
        return;
    }
    
    info!("MQTT Listener pripojeny na krystal/edge/status");

    tokio::spawn(async move {
        loop {
            match eventloop.poll().await {
                Ok(notification) => {
                    if let Event::Incoming(Incoming::Publish(p)) = notification {
                        if let Ok(payload) = serde_json::from_slice::<EdgeStatusPayload>(&p.payload) {
                            let mut prio = 0;
                            if payload.priority_task {
                                prio = payload.priority_level.unwrap_or(10); // default high priority
                                state.task_active.store(true, Ordering::SeqCst);
                            } else {
                                state.task_active.store(false, Ordering::SeqCst);
                            }
                            state.priority.store(prio, Ordering::SeqCst);
                            info!("MQTT Edge status update: Priority={}", prio);
                        }
                    }
                }
                Err(e) => {
                    error!("MQTT error: {}. Reconnecting...", e);
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
            }
        }
    });
}
