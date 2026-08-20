//! Runtime: Connects feature engine to live data sources
//! Fetches variables, computes features, and exposes functions

use crate::feature_engine::*;
use crate::types::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Variable source types
#[derive(Debug, Clone)]
pub enum VarSource {
    Telemetry,           // From TelemetrySnapshot
    Computed,            // Derived feature
    External(String),    // External API/file
    Constant(f64),       // Static value
    Cached { ttl_ms: u64 },
}

/// Runtime variable definition
#[derive(Debug, Clone)]
pub struct RuntimeVar {
    pub name: String,
    pub source: VarSource,
    pub expression: Option<String>,  // For computed vars
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// Function registry entry
#[derive(Clone)]
pub struct RuntimeFunc {
    pub name: String,
    pub arity: usize,
    pub func: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>,
}

impl std::fmt::Debug for RuntimeFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeFunc")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .finish()
    }
}

/// Cached value with expiration
#[derive(Debug, Clone)]
struct CachedValue {
    value: f64,
    expires_at: Instant,
}

/// Runtime context for feature evaluation
pub struct Runtime {
    feature_engine: FeatureEngine,
    variables: HashMap<String, RuntimeVar>,
    functions: HashMap<String, RuntimeFunc>,
    cache: RwLock<HashMap<String, CachedValue>>,
    telemetry: RwLock<Option<TelemetrySnapshot>>,
    budgets: RwLock<Option<ResourceBudgets>>,
}

impl Runtime {
    pub fn new() -> Self {
        let mut rt = Self {
            feature_engine: FeatureEngine::new(),
            variables: HashMap::new(),
            functions: HashMap::new(),
            cache: RwLock::new(HashMap::new()),
            telemetry: RwLock::new(None),
            budgets: RwLock::new(None),
        };
        rt.register_builtin_functions();
        rt.register_builtin_variables();
        rt
    }

    /// Register built-in functions
    fn register_builtin_functions(&mut self) {
        // Clamping
        self.register_function("clamp", 3, Arc::new(|args| {
            args[0].clamp(args[1], args[2])
        }));

        // Linear interpolation
        self.register_function("lerp", 3, Arc::new(|args| {
            args[0] + (args[1] - args[0]) * args[2]
        }));

        // Smoothstep
        self.register_function("smoothstep", 3, Arc::new(|args| {
            let t = ((args[0] - args[1]) / (args[2] - args[1])).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }));

        // Min/Max
        self.register_function("min", 2, Arc::new(|args| args[0].min(args[1])));
        self.register_function("max", 2, Arc::new(|args| args[0].max(args[1])));

        // Absolute difference
        self.register_function("absdiff", 2, Arc::new(|args| (args[0] - args[1]).abs()));

        // Normalize to range
        self.register_function("norm", 3, Arc::new(|args| {
            (args[0] - args[1]) / (args[2] - args[1]).max(1e-10)
        }));

        // Sigmoid
        self.register_function("sigmoid", 1, Arc::new(|args| {
            1.0 / (1.0 + (-args[0]).exp())
        }));

        // ReLU
        self.register_function("relu", 1, Arc::new(|args| args[0].max(0.0)));

        // Exponential moving average weight
        self.register_function("ema_weight", 1, Arc::new(|args| {
            2.0 / (args[0] + 1.0)
        }));
    }

    /// Register built-in variables from telemetry
    fn register_builtin_variables(&mut self) {
        let telemetry_vars = [
            ("cpu_util", 0.0, Some(0.0), Some(1.0)),
            ("gpu_util", 0.0, Some(0.0), Some(1.0)),
            ("frametime_ms", 16.67, Some(0.0), Some(1000.0)),
            ("temp_cpu", 50.0, Some(0.0), Some(120.0)),
            ("temp_gpu", 50.0, Some(0.0), Some(120.0)),
        ];

        for (name, default, min, max) in telemetry_vars {
            self.variables.insert(name.to_string(), RuntimeVar {
                name: name.to_string(),
                source: VarSource::Telemetry,
                expression: None,
                default,
                min,
                max,
            });
        }

        // Constants
        self.variables.insert("pi".to_string(), RuntimeVar {
            name: "pi".to_string(),
            source: VarSource::Constant(std::f64::consts::PI),
            expression: None,
            default: std::f64::consts::PI,
            min: None,
            max: None,
        });

        self.variables.insert("e".to_string(), RuntimeVar {
            name: "e".to_string(),
            source: VarSource::Constant(std::f64::consts::E),
            expression: None,
            default: std::f64::consts::E,
            min: None,
            max: None,
        });
    }

    /// Register a custom function
    pub fn register_function(
        &mut self,
        name: &str,
        arity: usize,
        func: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    ) {
        self.functions.insert(name.to_string(), RuntimeFunc {
            name: name.to_string(),
            arity,
            func,
        });
    }

    /// Register a computed variable
    pub fn register_computed_var(&mut self, name: &str, expression: &str, default: f64) {
        self.variables.insert(name.to_string(), RuntimeVar {
            name: name.to_string(),
            source: VarSource::Computed,
            expression: Some(expression.to_string()),
            default,
            min: None,
            max: None,
        });
    }

    /// Update telemetry snapshot
    pub fn update_telemetry(&self, snapshot: TelemetrySnapshot) {
        let mut tel = self.telemetry.write().unwrap();
        *tel = Some(snapshot);
    }

    /// Update resource budgets
    pub fn update_budgets(&self, budgets: ResourceBudgets) {
        let mut b = self.budgets.write().unwrap();
        *b = Some(budgets);
    }

    /// Fetch a variable value
    pub fn fetch_var(&self, name: &str) -> Option<f64> {
        let var_def = self.variables.get(name)?;

        // Check cache first
        if let VarSource::Cached { ttl_ms } = &var_def.source {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(name) {
                if cached.expires_at > Instant::now() {
                    return Some(cached.value);
                }
            }
        }

        let value = match &var_def.source {
            VarSource::Constant(v) => *v,
            VarSource::Telemetry => self.fetch_telemetry_var(name)?,
            VarSource::Computed => self.compute_var(var_def)?,
            VarSource::External(_path) => var_def.default, // TODO: implement
            VarSource::Cached { .. } => self.compute_var(var_def)?,
        };

        // Apply bounds
        let bounded = match (var_def.min, var_def.max) {
            (Some(min), Some(max)) => value.clamp(min, max),
            (Some(min), None) => value.max(min),
            (None, Some(max)) => value.min(max),
            (None, None) => value,
        };

        Some(bounded)
    }

    /// Fetch multiple variables
    pub fn fetch_vars(&self, names: &[&str]) -> HashMap<String, f64> {
        names.iter()
            .filter_map(|&name| self.fetch_var(name).map(|v| (name.to_string(), v)))
            .collect()
    }

    /// Fetch all registered variables
    pub fn fetch_all_vars(&self) -> HashMap<String, f64> {
        self.variables.keys()
            .filter_map(|name| self.fetch_var(name).map(|v| (name.clone(), v)))
            .collect()
    }

    fn fetch_telemetry_var(&self, name: &str) -> Option<f64> {
        let tel = self.telemetry.read().unwrap();
        let snapshot = tel.as_ref()?;

        Some(match name {
            "cpu_util" => snapshot.cpu_util,
            "gpu_util" => snapshot.gpu_util,
            "frametime_ms" => snapshot.frametime_ms,
            "temp_cpu" => snapshot.temp_cpu as f64,
            "temp_gpu" => snapshot.temp_gpu as f64,
            _ => return None,
        })
    }

    fn compute_var(&self, var: &RuntimeVar) -> Option<f64> {
        let expr_str = var.expression.as_ref()?;

        // Load current variables into engine
        let vars = self.fetch_base_vars();
        let mut engine = FeatureEngine::new();
        for (k, v) in vars {
            engine.set_variable(&k, v);
        }

        // Parse and evaluate
        let expr = engine.parse(expr_str).ok()?;
        engine.evaluate(&expr).ok()
    }

    fn fetch_base_vars(&self) -> HashMap<String, f64> {
        let mut vars = HashMap::new();

        if let Some(snapshot) = self.telemetry.read().unwrap().as_ref() {
            vars.insert("cpu_util".to_string(), snapshot.cpu_util);
            vars.insert("gpu_util".to_string(), snapshot.gpu_util);
            vars.insert("frametime_ms".to_string(), snapshot.frametime_ms);
            vars.insert("temp_cpu".to_string(), snapshot.temp_cpu as f64);
            vars.insert("temp_gpu".to_string(), snapshot.temp_gpu as f64);
        }

        vars.insert("pi".to_string(), std::f64::consts::PI);
        vars.insert("e".to_string(), std::f64::consts::E);

        vars
    }

    /// Call a registered function
    pub fn call_function(&self, name: &str, args: &[f64]) -> Option<f64> {
        let func = self.functions.get(name)?;
        if args.len() != func.arity {
            return None;
        }
        Some((func.func)(args))
    }

    /// Evaluate an expression with current runtime state
    pub fn evaluate(&self, expression: &str) -> Result<f64, String> {
        let vars = self.fetch_all_vars();
        let mut engine = FeatureEngine::new();
        for (k, v) in vars {
            engine.set_variable(&k, v);
        }
        let expr = engine.parse(expression)?;
        engine.evaluate(&expr)
    }

    /// Compute a feature with scaling
    pub fn compute_scaled_feature(
        &self,
        base_var: &str,
        params: &ScaleParams,
    ) -> Option<f64> {
        let value = self.fetch_var(base_var)?;
        Some(self.feature_engine.scale_abt(value, params))
    }

    /// Compute multiple features from definitions
    pub fn compute_features(&self, definitions: &[(String, String)]) -> HashMap<String, f64> {
        definitions.iter()
            .filter_map(|(name, expr)| {
                self.evaluate(expr).ok().map(|v| (name.clone(), v))
            })
            .collect()
    }

    /// List all registered variables
    pub fn list_variables(&self) -> Vec<&str> {
        self.variables.keys().map(|s| s.as_str()).collect()
    }

    /// List all registered functions
    pub fn list_functions(&self) -> Vec<(&str, usize)> {
        self.functions.iter().map(|(k, v)| (k.as_str(), v.arity)).collect()
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_telemetry() -> TelemetrySnapshot {
        TelemetrySnapshot {
            timestamp: "2025-01-21T14:30:00Z".into(),
            cpu_util: 0.75,
            gpu_util: 0.70,
            frametime_ms: 14.2,
            temp_cpu: 72,
            temp_gpu: 68,
            active_process_category: "gaming".into(),
        }
    }

    #[test]
    fn test_fetch_telemetry_var() {
        let rt = Runtime::new();
        rt.update_telemetry(sample_telemetry());

        assert!((rt.fetch_var("cpu_util").unwrap() - 0.75).abs() < 0.001);
        assert!((rt.fetch_var("temp_cpu").unwrap() - 72.0).abs() < 0.001);
    }

    #[test]
    fn test_computed_var() {
        let mut rt = Runtime::new();
        rt.update_telemetry(sample_telemetry());
        rt.register_computed_var("cpu_scaled", "cpu_util * 2.0", 0.0);

        assert!((rt.fetch_var("cpu_scaled").unwrap() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_call_function() {
        let rt = Runtime::new();

        assert!((rt.call_function("clamp", &[1.5, 0.0, 1.0]).unwrap() - 1.0).abs() < 0.001);
        assert!((rt.call_function("lerp", &[0.0, 10.0, 0.5]).unwrap() - 5.0).abs() < 0.001);
        assert!((rt.call_function("sigmoid", &[0.0]).unwrap() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_expression() {
        let rt = Runtime::new();
        rt.update_telemetry(sample_telemetry());

        let result = rt.evaluate("cpu_util + gpu_util").unwrap();
        assert!((result - 1.45).abs() < 0.001);
    }
}
