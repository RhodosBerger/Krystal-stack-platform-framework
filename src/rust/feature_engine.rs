//! Feature Engineering Engine with mathematical transformations
//! Supports alpha/beta/theta scaling, logarithmic features, and trigonometric parsing

use std::collections::HashMap;
use std::f64::consts::{E, PI};

/// Feature scaling parameters
#[derive(Debug, Clone)]
pub struct ScaleParams {
    pub alpha: f64,  // Primary scaling coefficient
    pub beta: f64,   // Secondary scaling coefficient
    pub theta: f64,  // Angular/phase parameter (radians)
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            theta: 0.0,
        }
    }
}

/// Logarithmic base options
#[derive(Debug, Clone, Copy)]
pub enum LogBase {
    Natural,    // ln (base e)
    Base2,      // log2
    Base10,     // log10
    Custom(f64),
}

/// Trigonometric function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrigFunc {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Asin,
    Acos,
    Atan,
}

/// Parsed mathematical expression
#[derive(Debug, Clone)]
pub enum MathExpr {
    Constant(f64),
    Variable(String),
    Scale { expr: Box<MathExpr>, params: ScaleParams },
    Log { expr: Box<MathExpr>, base: LogBase },
    Trig { func: TrigFunc, expr: Box<MathExpr> },
    BinaryOp { left: Box<MathExpr>, op: BinaryOp, right: Box<MathExpr> },
    Power { base: Box<MathExpr>, exp: Box<MathExpr> },
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Feature Engine for database-scale transformations
pub struct FeatureEngine {
    variables: HashMap<String, f64>,
    cache: HashMap<String, f64>,
}

impl FeatureEngine {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    pub fn set_variable(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), value);
        self.cache.clear();
    }

    pub fn set_variables(&mut self, vars: HashMap<String, f64>) {
        self.variables.extend(vars);
        self.cache.clear();
    }

    /// Apply alpha-beta-theta scaling: alpha * x + beta, with theta phase shift
    pub fn scale_abt(&self, value: f64, params: &ScaleParams) -> f64 {
        params.alpha * value + params.beta + params.theta.sin() * value.abs()
    }

    /// Apply logarithmic transformation
    pub fn log_transform(&self, value: f64, base: LogBase) -> f64 {
        let safe_val = value.abs().max(1e-10);
        match base {
            LogBase::Natural => safe_val.ln(),
            LogBase::Base2 => safe_val.log2(),
            LogBase::Base10 => safe_val.log10(),
            LogBase::Custom(b) => safe_val.ln() / b.ln(),
        }
    }

    /// Apply trigonometric function
    pub fn trig_transform(&self, value: f64, func: TrigFunc) -> f64 {
        match func {
            TrigFunc::Sin => value.sin(),
            TrigFunc::Cos => value.cos(),
            TrigFunc::Tan => value.tan(),
            TrigFunc::Sinh => value.sinh(),
            TrigFunc::Cosh => value.cosh(),
            TrigFunc::Tanh => value.tanh(),
            TrigFunc::Asin => value.clamp(-1.0, 1.0).asin(),
            TrigFunc::Acos => value.clamp(-1.0, 1.0).acos(),
            TrigFunc::Atan => value.atan(),
        }
    }

    /// Evaluate a parsed expression
    pub fn evaluate(&self, expr: &MathExpr) -> Result<f64, String> {
        match expr {
            MathExpr::Constant(v) => Ok(*v),

            MathExpr::Variable(name) => {
                self.variables.get(name)
                    .copied()
                    .ok_or_else(|| format!("Unknown variable: {}", name))
            }

            MathExpr::Scale { expr, params } => {
                let val = self.evaluate(expr)?;
                Ok(self.scale_abt(val, params))
            }

            MathExpr::Log { expr, base } => {
                let val = self.evaluate(expr)?;
                Ok(self.log_transform(val, *base))
            }

            MathExpr::Trig { func, expr } => {
                let val = self.evaluate(expr)?;
                Ok(self.trig_transform(val, *func))
            }

            MathExpr::BinaryOp { left, op, right } => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                Ok(match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => if r.abs() > 1e-10 { l / r } else { f64::NAN },
                })
            }

            MathExpr::Power { base, exp } => {
                let b = self.evaluate(base)?;
                let e = self.evaluate(exp)?;
                Ok(b.powf(e))
            }
        }
    }

    /// Parse a mathematical expression string
    pub fn parse(&self, input: &str) -> Result<MathExpr, String> {
        let input = input.trim();

        // Handle parentheses
        if input.starts_with('(') && input.ends_with(')') {
            return self.parse(&input[1..input.len()-1]);
        }

        // Try to parse as trig function
        for (name, func) in [
            ("sin", TrigFunc::Sin), ("cos", TrigFunc::Cos), ("tan", TrigFunc::Tan),
            ("sinh", TrigFunc::Sinh), ("cosh", TrigFunc::Cosh), ("tanh", TrigFunc::Tanh),
            ("asin", TrigFunc::Asin), ("acos", TrigFunc::Acos), ("atan", TrigFunc::Atan),
        ] {
            if input.starts_with(name) && input[name.len()..].starts_with('(') {
                let inner = self.extract_parens(&input[name.len()..])?;
                return Ok(MathExpr::Trig {
                    func,
                    expr: Box::new(self.parse(&inner)?),
                });
            }
        }

        // Try to parse as log function
        for (name, base) in [
            ("ln", LogBase::Natural), ("log2", LogBase::Base2), ("log10", LogBase::Base10), ("log", LogBase::Base10),
        ] {
            if input.starts_with(name) && input[name.len()..].starts_with('(') {
                let inner = self.extract_parens(&input[name.len()..])?;
                return Ok(MathExpr::Log {
                    base,
                    expr: Box::new(self.parse(&inner)?),
                });
            }
        }

        // Try to parse as scale function: scale(expr, alpha, beta, theta)
        if input.starts_with("scale(") {
            let inner = self.extract_parens(&input[5..])?;
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() >= 1 {
                let expr = self.parse(parts[0].trim())?;
                let alpha = parts.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(1.0);
                let beta = parts.get(2).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
                let theta = parts.get(3).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
                return Ok(MathExpr::Scale {
                    expr: Box::new(expr),
                    params: ScaleParams { alpha, beta, theta },
                });
            }
        }

        // Try binary operators (lowest precedence first)
        for (op_char, op) in [('+', BinaryOp::Add), ('-', BinaryOp::Sub)] {
            if let Some(pos) = self.find_operator(input, op_char) {
                return Ok(MathExpr::BinaryOp {
                    left: Box::new(self.parse(&input[..pos])?),
                    op,
                    right: Box::new(self.parse(&input[pos+1..])?),
                });
            }
        }

        for (op_char, op) in [('*', BinaryOp::Mul), ('/', BinaryOp::Div)] {
            if let Some(pos) = self.find_operator(input, op_char) {
                return Ok(MathExpr::BinaryOp {
                    left: Box::new(self.parse(&input[..pos])?),
                    op,
                    right: Box::new(self.parse(&input[pos+1..])?),
                });
            }
        }

        // Try power operator
        if let Some(pos) = self.find_operator(input, '^') {
            return Ok(MathExpr::Power {
                base: Box::new(self.parse(&input[..pos])?),
                exp: Box::new(self.parse(&input[pos+1..])?),
            });
        }

        // Try to parse as constant
        if let Ok(v) = input.parse::<f64>() {
            return Ok(MathExpr::Constant(v));
        }

        // Constants
        match input.to_lowercase().as_str() {
            "pi" => return Ok(MathExpr::Constant(PI)),
            "e" => return Ok(MathExpr::Constant(E)),
            _ => {}
        }

        // Must be a variable
        if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Ok(MathExpr::Variable(input.to_string()));
        }

        Err(format!("Cannot parse: {}", input))
    }

    fn extract_parens(&self, input: &str) -> Result<String, String> {
        if !input.starts_with('(') {
            return Err("Expected '('".into());
        }
        let mut depth = 0;
        for (i, c) in input.chars().enumerate() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(input[1..i].to_string());
                    }
                }
                _ => {}
            }
        }
        Err("Unmatched parentheses".into())
    }

    fn find_operator(&self, input: &str, op: char) -> Option<usize> {
        let mut depth = 0;
        for (i, c) in input.chars().enumerate().rev() {
            match c {
                ')' => depth += 1,
                '(' => depth -= 1,
                ch if ch == op && depth == 0 && i > 0 => return Some(i),
                _ => {}
            }
        }
        None
    }
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Database-scale feature transformer
pub struct DbFeatureTransformer {
    engine: FeatureEngine,
    feature_definitions: HashMap<String, String>, // name -> expression
}

impl DbFeatureTransformer {
    pub fn new() -> Self {
        Self {
            engine: FeatureEngine::new(),
            feature_definitions: HashMap::new(),
        }
    }

    /// Define a new feature with a mathematical expression
    pub fn define_feature(&mut self, name: &str, expression: &str) {
        self.feature_definitions.insert(name.to_string(), expression.to_string());
    }

    /// Transform a batch of records
    pub fn transform_batch(&mut self, records: &[HashMap<String, f64>]) -> Vec<HashMap<String, f64>> {
        records.iter().map(|record| self.transform_record(record)).collect()
    }

    /// Transform a single record, computing all defined features
    pub fn transform_record(&mut self, record: &HashMap<String, f64>) -> HashMap<String, f64> {
        self.engine.set_variables(record.clone());

        let mut result = record.clone();

        for (name, expr_str) in &self.feature_definitions {
            if let Ok(expr) = self.engine.parse(expr_str) {
                if let Ok(value) = self.engine.evaluate(&expr) {
                    result.insert(name.clone(), value);
                }
            }
        }

        result
    }

    /// Add standard scaling features
    pub fn add_standard_features(&mut self, base_col: &str) {
        // Log transforms
        self.define_feature(&format!("{}_log", base_col), &format!("ln({})", base_col));
        self.define_feature(&format!("{}_log10", base_col), &format!("log10({})", base_col));

        // Trig transforms (for cyclical features)
        self.define_feature(&format!("{}_sin", base_col), &format!("sin({})", base_col));
        self.define_feature(&format!("{}_cos", base_col), &format!("cos({})", base_col));

        // Alpha-beta-theta scaled
        self.define_feature(&format!("{}_scaled", base_col), &format!("scale({}, 1.0, 0.0, 0.0)", base_col));
    }
}

impl Default for DbFeatureTransformer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_abt() {
        let engine = FeatureEngine::new();
        let params = ScaleParams { alpha: 2.0, beta: 1.0, theta: 0.0 };
        assert!((engine.scale_abt(5.0, &params) - 11.0).abs() < 0.001);
    }

    #[test]
    fn test_log_transform() {
        let engine = FeatureEngine::new();
        assert!((engine.log_transform(E, LogBase::Natural) - 1.0).abs() < 0.001);
        assert!((engine.log_transform(100.0, LogBase::Base10) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_trig_transform() {
        let engine = FeatureEngine::new();
        assert!((engine.trig_transform(0.0, TrigFunc::Sin) - 0.0).abs() < 0.001);
        assert!((engine.trig_transform(0.0, TrigFunc::Cos) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_and_evaluate() {
        let mut engine = FeatureEngine::new();
        engine.set_variable("x", 2.0);

        let expr = engine.parse("sin(x)").unwrap();
        let result = engine.evaluate(&expr).unwrap();
        assert!((result - 2.0_f64.sin()).abs() < 0.001);

        let expr = engine.parse("ln(x)").unwrap();
        let result = engine.evaluate(&expr).unwrap();
        assert!((result - 2.0_f64.ln()).abs() < 0.001);

        let expr = engine.parse("scale(x, 2.0, 1.0, 0.0)").unwrap();
        let result = engine.evaluate(&expr).unwrap();
        assert!((result - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_binary_ops() {
        let mut engine = FeatureEngine::new();
        engine.set_variable("x", 3.0);
        engine.set_variable("y", 4.0);

        let expr = engine.parse("x + y").unwrap();
        assert!((engine.evaluate(&expr).unwrap() - 7.0).abs() < 0.001);

        let expr = engine.parse("x * y").unwrap();
        assert!((engine.evaluate(&expr).unwrap() - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_db_transformer() {
        let mut transformer = DbFeatureTransformer::new();
        transformer.define_feature("x_squared", "x^2");
        transformer.define_feature("x_sin", "sin(x)");

        let record: HashMap<String, f64> = [("x".to_string(), PI)].into_iter().collect();
        let result = transformer.transform_record(&record);

        assert!((result["x_squared"] - PI * PI).abs() < 0.001);
        assert!(result["x_sin"].abs() < 0.001); // sin(PI) ≈ 0
    }
}
