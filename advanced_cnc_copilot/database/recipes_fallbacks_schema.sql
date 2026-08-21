-- ============================================================================
-- Production Recipes & Fallback Alternatives Schema
-- Stores alternative approaches when primary methods fail
-- ============================================================================

-- Production Recipes table
CREATE TABLE IF NOT EXISTS production_recipes (
    id SERIAL PRIMARY KEY,
    recipe_id VARCHAR(50) UNIQUE NOT NULL,
    recipe_name VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,  -- 'communication', 'processing', 'data_access', 'ui', 'integration'
    
    -- Primary method
    primary_method VARCHAR(200) NOT NULL,
    primary_description TEXT,
    primary_requirements JSONB,  -- Dependencies, conditions
    
    -- Alternative method
    alternative_method VARCHAR(200) NOT NULL,
    alternative_description TEXT,
    alternative_requirements JSONB,
    
    -- Trigger conditions
    fallback_triggers TEXT[],  -- When to use alternative
    trigger_conditions JSONB,
    
    -- Effectiveness
    confidence_score DECIMAL(3,2) CHECK (confidence_score BETWEEN 0 AND 1),
    success_rate DECIMAL(3,2),
    times_used INTEGER DEFAULT 0,
    times_successful INTEGER DEFAULT 0,
    
    -- Usage statistics
    average_response_time_ms INTEGER,
    resource_usage VARCHAR(50),  -- 'low', 'medium', 'high'
    
    -- Metadata
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    enabled BOOLEAN DEFAULT TRUE,
    priority_order INTEGER DEFAULT 1,
    
    -- Documentation
    usage_instructions TEXT,
    limitations TEXT,
    best_practices TEXT
);

CREATE INDEX idx_recipes_category ON production_recipes(category);
CREATE INDEX idx_recipes_enabled ON production_recipes(enabled);
CREATE INDEX idx_recipes_success_rate ON production_recipes(success_rate DESC);


-- Recipe Execution Log
CREATE TABLE IF NOT EXISTS recipe_executions (
    id SERIAL PRIMARY KEY,
    execution_id VARCHAR(50) UNIQUE NOT NULL,
    recipe_id VARCHAR(50) REFERENCES production_recipes(recipe_id),
    
    -- Execution context
    triggered_by VARCHAR(50),  -- 'system', 'user', 'automation'
    trigger_reason TEXT,
    execution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Attempt details
    primary_attempted BOOLEAN DEFAULT TRUE,
    primary_success BOOLEAN,
    primary_error_message TEXT,
    primary_duration_ms INTEGER,
    
    fallback_attempted BOOLEAN DEFAULT FALSE,
    fallback_success BOOLEAN,
    fallback_error_message TEXT,
    fallback_duration_ms INTEGER,
    
    -- Results
    final_result VARCHAR(20),  -- 'success', 'failed', 'partial'
    output_data JSONB,
    
    -- User feedback
    user_satisfaction INTEGER CHECK (user_satisfaction BETWEEN 1 AND 5),
    user_feedback TEXT
);

CREATE INDEX idx_executions_recipe ON recipe_executions(recipe_id);
CREATE INDEX idx_executions_timestamp ON recipe_executions(execution_timestamp);
CREATE INDEX idx_executions_result ON recipe_executions(final_result);


-- Equivalent Communication Patterns table
CREATE TABLE IF NOT EXISTS communication_patterns (
    id SERIAL PRIMARY KEY,
    pattern_id VARCHAR(50) UNIQUE NOT NULL,
    pattern_name VARCHAR(200) NOT NULL,
    
    -- Pattern details
    intent VARCHAR(50),  -- Links to conversation_mediator intents
    topic VARCHAR(50),
    
    -- Primary communication style
    primary_format VARCHAR(50),  -- 'json', 'natural_language', 'structured_data', 'visual'
    primary_template TEXT,
   primary_example TEXT,
    
    -- Alternative formats
    alternative_format VARCHAR(50),
    alternative_template TEXT,
    alternative_example TEXT,
    
    -- When to use alternative
    use_alternative_when TEXT[],  -- Conditions
    user_preferences JSONB,  -- User-specific preferences
    
    -- Effectiveness metrics
    clarity_score DECIMAL(3,2),
    user_preference_score DECIMAL(3,2),
    times_used INTEGER DEFAULT 0,
    
    -- Context
    applicable_roles TEXT[],  -- 'operator', 'engineer', 'manager'
    applicable_scenarios TEXT[],
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patterns_intent ON communication_patterns(intent);
CREATE INDEX idx_patterns_topic ON communication_patterns(topic);


-- Theme Preferences table
CREATE TABLE IF NOT EXISTS user_theme_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    
    -- Selected theme
    selected_theme_id VARCHAR(50) NOT NULL,
    theme_name VARCHAR(100),
    
    -- How selected
    selection_method VARCHAR(50),  -- 'manual', 'llm_suggested', 'default'
    llm_suggested_theme VARCHAR(50),
    llm_reasoning TEXT,
    llm_confidence DECIMAL(3,2),
    
    -- User context at time of selection
    user_role VARCHAR(50),
    shift_type VARCHAR(20),  -- 'day', 'night', 'swing'
    workload_level VARCHAR(20),
    
    -- Theme effectiveness
    switch_count INTEGER DEFAULT 0,  -- How many times changed
    session_duration_minutes INTEGER,
    user_satisfaction INTEGER CHECK (user_satisfaction BETWEEN 1 AND 5),
    
    -- Timestamps
    selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    
    UNIQUE(user_id, selected_theme_id)
);

CREATE INDEX idx_theme_prefs_user ON user_theme_preferences(user_id);
CREATE INDEX idx_theme_prefs_theme ON user_theme_preferences(selected_theme_id);


-- LLM Suggestion Log
CREATE TABLE IF NOT EXISTS llm_suggestions_log (
    id SERIAL PRIMARY KEY,
    suggestion_id VARCHAR(50) UNIQUE NOT NULL,
    
    -- What was suggested
    suggestion_type VARCHAR(50),  -- 'theme', 'recipe', 'communication', 'optimization'
    suggested_item VARCHAR(200),
    confidence_score DECIMAL(3,2),
    
    -- Reasoning
    llm_reasoning TEXT NOT NULL,
    factors_considered JSONB,
    
    -- User context
    user_id VARCHAR(50),
    user_role VARCHAR(50),
    context_data JSONB,
    
    -- Outcome
    user_accepted BOOLEAN,
    user_feedback TEXT,
    alternative_chosen VARCHAR(200),
    
    -- Model info
    llm_model VARCHAR(100),
    prompt_template VARCHAR(50),
    response_time_ms INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_suggestions_type ON llm_suggestions_log(suggestion_type);
CREATE INDEX idx_suggestions_accepted ON llm_suggestions_log(user_accepted);


-- ============================================================================
-- SAMPLE DATA
-- ============================================================================

-- Insert sample production recipes
INSERT INTO production_recipes (
    recipe_id, recipe_name, category, 
    primary_method, primary_description, alternative_method, alternative_description,
    fallback_triggers, confidence_score, success_rate, usage_instructions
) VALUES 
(
    'FALLBACK-001',
    'LLM API Timeout Handling',
    'communication',
    'OpenAI GPT-4 API',
    'Primary cloud-based LLM for all AI features',
    'Local Llama 2 Model',
    'Locally hosted LLM that runs without internet connectivity',
    ARRAY['api_timeout', 'connection_error', 'rate_limit', 'service_unavailable'],
    0.92,
    0.88,
    'Automatically switch to local model after 3 failed API attempts or 5-second timeout. Local model provides 80% accuracy of cloud model but with guaranteed availability.'
),
(
    'FALLBACK-002',
    'Database Connection Recovery',
    'data_access',
    'PostgreSQL Primary Connection',
    'Main database connection for all read/write operations',
    'Read-only Replica + Redis Cache',
    'Fallback to replica database with cached recent data',
    ARRAY['connection_lost', 'timeout', 'network_error'],
    0.88,
    0.91,
    'On primary database failure, switch to read-only replica for queries. Use Redis cache for recent data. Display notice to user that live updates are paused. Attempt reconnection every 30 seconds.'
),
(
    'FALLBACK-003',
    'Real-time Sensor Data Unavailable',
    'processing',
    'Live MQTT Sensor Stream',
    'Real-time sensor data from shop floor machines',
    'Historical Pattern Simulation',
    'Generate simulated data based on historical patterns for this operation',
    ARRAY['mqtt_disconnected', 'sensor_offline', 'network_partition'],
    0.75,
    0.82,
    'When real-time data unavailable, use last 7 days of historical data to generate realistic simulated values. Clearly mark all data as "SIMULATED" in dashboard. Alert maintenance team of sensor issues.'
),
(
    'FALLBACK-004',
    'G-Code Validation Failure',
    'processing',
    'Advanced Syntax + Safety Validation',
    'Full G-Code parsing with collision detection and simulation',
    'Basic Safety Checks + Manual Review',
    'Minimal validation checking only for critical safety issues',
    ARRAY['parser_error', 'complex_syntax', 'unknown_codes'],
    0.85,
    0.79,
    'If advanced validation fails, run basic safety checks (spindle speed limits, axis limits, emergency stops present). Flag program for manual operator review before execution. Log validation failure for engineering review.'
),
(
    'FALLBACK-005',
    'Natural Language Understanding Failure',
    'communication',
    'Advanced NLP with Context',
    'Full natural language processing with conversation context',
    'Keyword Matching + Clarifying Questions',
    'Simple keyword-based matching with follow-up questions to user',
    ARRAY['low_confidence', 'ambiguous_input', 'unknown_intent'],
    0.71,
    0.76,
    'When NLP confidence < 70%, fall back to keyword matching and ask clarifying questions. Provide multiple choice options to user. Learn from user selection to improve future understanding.'
);


-- Insert sample communication patterns
INSERT INTO communication_patterns (
    pattern_id, pattern_name, intent, topic,
    primary_format, primary_template, alternative_format, alternative_template,
    use_alternative_when, applicable_roles
) VALUES 
(
    'COMM-001',
    'Technical Response - Speeds & Feeds',
    'technical_question',
    'speeds_feeds',
    'structured_data',
    'For {material} using {tool}: Speed {speed} RPM, Feed {feed} mm/min, DOC {doc} mm',
    'natural_language',
    'I recommend running your {tool} at {speed} RPM with a feed rate of {feed} mm/min. Take cuts of {doc} mm depth. This will give you a good balance of speed and tool life for {material}.',
    ARRAY['user_prefers_conversational', 'low_technical_expertise', 'training_mode'],
    ARRAY['operator', 'apprentice']
),
(
    'COMM-002',
    'Quote Response',
    'quote_request',
    'cost_estimation',
    'json',
    '{"unit_price": {price}, "total": {total}, "lead_time_days": {days}}',
    'natural_language',
    'Great! I can make those {parts} for ${price} each, which comes to ${total} total for {quantity} units. We can have them ready in {days} days.',
    ARRAY['customer_facing', 'sales_context', 'informal_request'],
    ARRAY['sales', 'customer']
),
(
    'COMM-003',
    'Error Messages',
    'troubleshooting',
    'quality_control',
    'technical',
    'ERROR: Dimensional tolerance exceeded. Part #{id}: {dimension} = {value} mm (tolerance ±{tolerance} mm)',
    'plain_language',
    'Heads up - Part #{id} is slightly out of spec. The {dimension} measurement came in at {value} mm, but it needs to be within ±{tolerance} mm. Let''s check the setup and try again.',
    ARRAY['operator_stress_high', 'urgent_context', 'first_time_user'],
    ARRAY['operator']
);


-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to get best recipe for scenario
CREATE OR REPLACE FUNCTION get_fallback_recipe(
    p_scenario VARCHAR(50)
) RETURNS TABLE (
    recipe_id VARCHAR(50),
    recipe_name VARCHAR(200),
    alternative_method VARCHAR(200),
    instructions TEXT,
    confidence DECIMAL(3,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pr.recipe_id,
        pr.recipe_name,
        pr.alternative_method,
        pr.usage_instructions,
        pr.confidence_score
    FROM production_recipes pr
    WHERE p_scenario = ANY(pr.fallback_triggers)
      AND pr.enabled = TRUE
    ORDER BY pr.confidence_score DESC, pr.success_rate DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;


-- Function to log recipe execution
CREATE OR REPLACE FUNCTION log_recipe_execution(
    p_recipe_id VARCHAR(50),
    p_primary_success BOOLEAN,
    p_fallback_success BOOLEAN,
    p_trigger_reason TEXT
) RETURNS VARCHAR(50) AS $$
DECLARE
    v_execution_id VARCHAR(50);
    v_final_result VARCHAR(20);
BEGIN
    -- Generate execution ID
    v_execution_id := 'EXEC-' || to_char(NOW(), 'YYYYMMDDHH24MISS');
    
    -- Determine final result
    IF p_primary_success THEN
        v_final_result := 'success';
    ELSIF p_fallback_success THEN
        v_final_result := 'success';
    ELSE
        v_final_result := 'failed';
    END IF;
    
    -- Insert execution log
    INSERT INTO recipe_executions (
        execution_id, recipe_id, trigger_reason,
        primary_attempted, primary_success,
        fallback_attempted, fallback_success,
        final_result
    ) VALUES (
        v_execution_id, p_recipe_id, p_trigger_reason,
        TRUE, p_primary_success,
        NOT p_primary_success, p_fallback_success,
        v_final_result
    );
    
    -- Update recipe statistics
    UPDATE production_recipes
    SET times_used = times_used + 1,
        times_successful = times_successful + CASE WHEN v_final_result = 'success' THEN 1 ELSE 0 END,
        success_rate = (times_successful + CASE WHEN v_final_result = 'success' THEN 1 ELSE 0 END)::DECIMAL / (times_used + 1),
        last_used = NOW()
    WHERE recipe_id = p_recipe_id;
    
    RETURN v_execution_id;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- VIEWS
-- ============================================================================

-- Most reliable fallback recipes
CREATE OR REPLACE VIEW v_top_fallback_recipes AS
SELECT
    recipe_id,
    recipe_name,
    category,
    primary_method,
    alternative_method,
    confidence_score,
    success_rate,
    times_used,
    ROUND((confidence_score + success_rate) / 2, 3) AS combined_score
FROM production_recipes
WHERE enabled = TRUE
  AND times_used >= 5
ORDER BY combined_score DESC
LIMIT 10;


-- Recent recipe executions with success rate
CREATE OR REPLACE VIEW v_recent_fallback_usage AS
SELECT
    pr.recipe_name,
    pr.category,
    COUNT(*) AS total_executions,
    SUM(CASE WHEN re.final_result = 'success' THEN 1 ELSE 0 END) AS successful,
    ROUND(SUM(CASE WHEN re.final_result = 'success' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) * 100, 1) AS success_percentage,
    MAX(re.execution_timestamp) AS last_used
FROM production_recipes pr
JOIN recipe_executions re ON pr.recipe_id = re.recipe_id
WHERE re.execution_timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY pr.recipe_name, pr.category
ORDER BY total_executions DESC;


-- LLM suggestion acceptance rate
CREATE OR REPLACE VIEW v_llm_suggestion_performance AS
SELECT
    suggestion_type,
    COUNT(*) AS total_suggestions,
    SUM(CASE WHEN user_accepted THEN 1 ELSE 0 END) AS accepted,
    ROUND(SUM(CASE WHEN user_accepted THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) * 100, 1) AS acceptance_rate,
    ROUND(AVG(confidence_score), 3) AS avg_confidence
FROM llm_suggestions_log
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY suggestion_type
ORDER BY acceptance_rate DESC;


-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Get fallback for specific scenario
-- SELECT * FROM get_fallback_recipe('api_timeout');

-- Log a recipe execution
-- SELECT log_recipe_execution('FALLBACK-001', FALSE, TRUE, 'OpenAI API timeout after 5 seconds');

-- View most reliable recipes
-- SELECT * FROM v_top_fallback_recipes;

-- Recent fallback usage
-- SELECT * FROM v_recent_fallback_usage;

-- LLM suggestion performance
-- SELECT * FROM v_llm_suggestion_performance;
