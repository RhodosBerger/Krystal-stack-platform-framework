//! Dependency Injection Framework and Feature Abstraction Layer
//! 
//! This module provides a flexible architecture for inserting new features
//! into the Krystal Stack without modifying core components.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Feature trait that all plugins must implement
pub trait Feature: Send + Sync {
    /// Unique identifier for the feature
    fn id(&self) -> &str;
    
    /// Feature name (human-readable)
    fn name(&self) -> &str;
    
    /// Feature version
    fn version(&self) -> &str;
    
    /// Initialize the feature
    fn initialize(&mut self) -> Result<(), String>;
    
    /// Shutdown the feature
    fn shutdown(&mut self) -> Result<(), String>;
    
    /// Get feature configuration
    fn get_config(&self) -> HashMap<String, String>;
    
    /// Set feature configuration
    fn set_config(&mut self, key: &str, value: &str) -> Result<(), String>;
    
    /// Execute the feature's main logic
    fn execute(&self, context: &FeatureContext) -> Result<FeatureResult, String>;
    
    /// Get feature dependencies
    fn dependencies(&self) -> Vec<&str>;
    
    /// Priority level (higher = executed first)
    fn priority(&self) -> u8 {
        128
    }
}

/// Feature execution context
#[derive(Clone, Debug)]
pub struct FeatureContext {
    pub data: Arc<DashMap<String, Arc<dyn Any + Send + Sync>>>,
    pub metadata: HashMap<String, String>,
}

impl FeatureContext {
    pub fn new() -> Self {
        FeatureContext {
            data: Arc::new(DashMap::new()),
            metadata: HashMap::new(),
        }
    }
    
    pub fn insert<T: Any + Send + Sync>(&self, key: String, value: T) {
        self.data.insert(key, Arc::new(value));
    }
    
    pub fn get<T: Any + Send + Sync>(&self, key: &str) -> Option<Arc<T>> {
        self.data.get(key).and_then(|v| v.clone().downcast().ok())
    }
    
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

impl Default for FeatureContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureResult {
    pub success: bool,
    pub data: Option<String>,
    pub metrics: HashMap<String, f64>,
    pub execution_time_ns: u128,
}

impl FeatureResult {
    pub fn success() -> Self {
        FeatureResult {
            success: true,
            data: None,
            metrics: HashMap::new(),
            execution_time_ns: 0,
        }
    }
    
    pub fn with_data(data: String) -> Self {
        FeatureResult {
            success: true,
            data: Some(data),
            metrics: HashMap::new(),
            execution_time_ns: 0,
        }
    }
    
    pub fn failure() -> Self {
        FeatureResult {
            success: false,
            data: None,
            metrics: HashMap::new(),
            execution_time_ns: 0,
        }
    }
}

/// Feature descriptor for Python interop
#[derive(Clone, Serialize, Deserialize)]
pub struct FeatureDescriptor {
    pub id: String,
    pub name: String,
    pub version: String,
    pub priority: u8,
    pub dependencies: Vec<String>,
    pub config: HashMap<String, String>,
    pub is_active: bool,
}

/// Container for managing features
#[pyclass]
pub struct FeatureContainer {
    features: Arc<DashMap<String, Box<dyn Feature>>>,
    feature_order: Arc<DashMap<String, u8>>,
    context: Arc<FeatureContext>,
    initialized: std::sync::atomic::AtomicBool,
}

#[pymethods]
impl FeatureContainer {
    #[new]
    fn new() -> Self {
        FeatureContainer {
            features: Arc::new(DashMap::new()),
            feature_order: Arc::new(DashMap::new()),
            context: Arc::new(FeatureContext::new()),
            initialized: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Register a feature (Python callback-based)
    #[pyo3(text_signature = "($self, feature_id, name, version, priority=128, dependencies=None)")]
    fn register_feature(&self, feature_id: String, name: String, version: String, priority: u8, dependencies: Option<Vec<String>>) {
        // This is a placeholder - in practice, Python would provide feature implementations
        // via callbacks or the features would be implemented in Rust
        self.feature_order.insert(feature_id, priority);
    }

    /// Get list of registered features
    #[pyo3(text_signature = "($self)")]
    fn list_features(&self) -> Vec<Py<PyAny>> {
        Python::with_gil(|py| {
            self.feature_order.iter().map(|r| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("id", r.key().clone()).ok();
                dict.set_item("priority", r.value()).ok();
                dict.into()
            }).collect()
        })
    }

    /// Initialize all features
    fn initialize_all(&self) -> bool {
        // Sort features by priority and initialize
        let mut features: Vec<_> = self.feature_order.iter().collect();
        features.sort_by(|a, b| b.value().cmp(a.value()));
        
        for entry in features {
            // Initialize in priority order
            // In a full implementation, this would call feature.initialize()
        }
        
        self.initialized.store(true, std::sync::atomic::Ordering::SeqCst);
        true
    }

    /// Execute all features
    #[pyo3(text_signature = "($self)")]
    fn execute_all(&self) -> Vec<Py<PyAny>> {
        Python::with_gil(|py| {
            let mut results = Vec::new();
            
            // Execute in priority order
            let mut features: Vec<_> = self.feature_order.iter().collect();
            features.sort_by(|a, b| b.value().cmp(a.value()));
            
            for entry in features {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("feature_id", entry.key().clone()).ok();
                dict.set_item("success", true).ok();
                results.push(dict.into());
            }
            
            results
        })
    }

    /// Get feature by ID
    #[pyo3(text_signature = "($self, feature_id)")]
    fn get_feature(&self, feature_id: String) -> Option<Py<PyAny>> {
        if self.feature_order.contains_key(&feature_id) {
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("id", feature_id).ok();
                Some(dict.into())
            })
        } else {
            None
        }
    }

    /// Remove a feature
    fn remove_feature(&self, feature_id: String) -> bool {
        self.feature_order.remove(&feature_id).is_some()
    }

    /// Check if container is initialized
    fn is_initialized(&self) -> bool {
        self.initialized.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Set context data
    #[pyo3(text_signature = "($self, key, value)")]
    fn set_context_data(&self, key: String, value: String) {
        self.context.set_metadata(key, value);
    }

    /// Get context data
    fn get_context_data(&self, key: String) -> Option<String> {
        self.context.get_metadata(&key).cloned()
    }
}

/// Module registry for dynamic feature loading
#[pyclass]
pub struct ModuleRegistry {
    modules: Arc<DashMap<String, ModuleInfo>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub version: String,
    pub path: String,
    pub features: Vec<String>,
    pub is_loaded: bool,
}

#[pymethods]
impl ModuleRegistry {
    #[new]
    fn new() -> Self {
        ModuleRegistry {
            modules: Arc::new(DashMap::new()),
        }
    }

    #[pyo3(text_signature = "($self, name, version, path, features=None)")]
    fn register_module(&self, name: String, version: String, path: String, features: Option<Vec<String>>) {
        let info = ModuleInfo {
            name: name.clone(),
            version,
            path,
            features: features.unwrap_or_default(),
            is_loaded: false,
        };
        self.modules.insert(name, info);
    }

    #[pyo3(text_signature = "($self, name)")]
    fn load_module(&self, name: String) -> bool {
        if let Some(mut info) = self.modules.get_mut(&name) {
            info.is_loaded = true;
            true
        } else {
            false
        }
    }

    fn unload_module(&self, name: String) -> bool {
        if let Some(mut info) = self.modules.get_mut(&name) {
            info.is_loaded = false;
            true
        } else {
            false
        }
    }

    #[pyo3(text_signature = "($self)")]
    fn list_modules(&self) -> Vec<Py<PyAny>> {
        Python::with_gil(|py| {
            self.modules.iter().map(|r| {
                let dict = pyo3::types::PyDict::new(py);
                let info = r.value();
                dict.set_item("name", info.name.clone()).ok();
                dict.set_item("version", info.version.clone()).ok();
                dict.set_item("path", info.path.clone()).ok();
                dict.set_item("features", info.features.clone()).ok();
                dict.set_item("is_loaded", info.is_loaded).ok();
                dict.into()
            }).collect()
        })
    }

    #[pyo3(text_signature = "($self, name)")]
    fn get_module(&self, name: String) -> Option<Py<PyAny>> {
        Python::with_gil(|py| {
            if let Some(info) = self.modules.get(&name) {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("name", info.name.clone()).ok();
                dict.set_item("version", info.version.clone()).ok();
                dict.set_item("path", info.path.clone()).ok();
                dict.set_item("features", info.features.clone()).ok();
                dict.set_item("is_loaded", info.is_loaded).ok();
                Some(dict.into())
            } else {
                None
            }
        })
    }
}

/// Hot reload manager for dynamic feature updates
#[pyclass]
pub struct HotReloadManager {
    watched_modules: Arc<DashMap<String, ModuleWatcher>>,
}

#[derive(Clone)]
struct ModuleWatcher {
    last_modified: u64,
    reload_count: u32,
}

#[pymethods]
impl HotReloadManager {
    #[new]
    fn new() -> Self {
        HotReloadManager {
            watched_modules: Arc::new(DashMap::new()),
        }
    }

    #[pyo3(text_signature = "($self, module_name, path)")]
    fn watch_module(&self, module_name: String, path: String) {
        // In a full implementation, this would set up file system watchers
        self.watched_modules.insert(module_name, ModuleWatcher {
            last_modified: 0,
            reload_count: 0,
        });
    }

    fn unwatch_module(&self, module_name: String) -> bool {
        self.watched_modules.remove(&module_name).is_some()
    }

    #[pyo3(text_signature = "($self)")]
    fn check_for_changes(&self) -> Vec<String> {
        // In a full implementation, this would check file modification times
        Vec::new()
    }

    #[pyo3(text_signature = "($self, module_name)")]
    fn trigger_reload(&self, module_name: String) -> bool {
        if let Some(mut watcher) = self.watched_modules.get_mut(&module_name) {
            watcher.reload_count += 1;
            true
        } else {
            false
        }
    }

    #[pyo3(text_signature = "($self, module_name)")]
    fn get_reload_count(&self, module_name: String) -> u32 {
        self.watched_modules
            .get(&module_name)
            .map(|w| w.reload_count)
            .unwrap_or(0)
    }
}

/// Builder pattern for constructing feature pipelines
pub struct FeaturePipelineBuilder {
    features: Vec<String>,
    context: FeatureContext,
}

impl FeaturePipelineBuilder {
    pub fn new() -> Self {
        FeaturePipelineBuilder {
            features: Vec::new(),
            context: FeatureContext::new(),
        }
    }
    
    pub fn add_feature(mut self, feature_id: String) -> Self {
        self.features.push(feature_id);
        self
    }
    
    pub fn with_context_data(mut self, key: String, value: String) -> Self {
        self.context.set_metadata(key, value);
        self
    }
    
    pub fn build(self) -> FeaturePipeline {
        FeaturePipeline {
            features: self.features,
            context: Arc::new(self.context),
        }
    }
}

impl Default for FeaturePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature pipeline for executing features in sequence
pub struct FeaturePipeline {
    features: Vec<String>,
    context: Arc<FeatureContext>,
}

impl FeaturePipeline {
    pub fn execute(&self) -> Vec<FeatureResult> {
        self.features.iter().map(|_| {
            // In a full implementation, this would execute each feature
            FeatureResult::success()
        }).collect()
    }
    
    pub fn get_context(&self) -> Arc<FeatureContext> {
        self.context.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_container() {
        let container = FeatureContainer::new();
        container.register_feature(
            "test_feature".to_string(),
            "Test Feature".to_string(),
            "1.0.0".to_string(),
            100,
            Some(vec![])
        );
        
        assert!(container.list_features().len() > 0);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = FeaturePipelineBuilder::new()
            .add_feature("feature1".to_string())
            .add_feature("feature2".to_string())
            .with_context_data("key".to_string(), "value".to_string())
            .build();
        
        assert_eq!(pipeline.features.len(), 2);
    }
}
