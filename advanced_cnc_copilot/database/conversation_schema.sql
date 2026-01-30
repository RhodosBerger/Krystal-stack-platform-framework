-- ============================================================================
-- Conversation Logging &  Topic Relationship Schema
-- Stores all LLM interactions and builds knowledge graph
-- ============================================================================

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(50) UNIQUE NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP,
    total_turns INTEGER DEFAULT 0,
    primary_intent VARCHAR(50),
    active_topics TEXT[],  -- Array of topic strings
    satisfaction_rating INTEGER CHECK (satisfaction_rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversations_user ON conversations(user_id);
CREATE INDEX idx_conversations_start ON conversations(session_start);


-- Conversation Turns table
CREATE TABLE IF NOT EXISTS conversation_turns (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(50) REFERENCES conversations(conversation_id),
    turn_number INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- User input
    user_input TEXT NOT NULL,
    user_intent VARCHAR(50),
    intent_confidence DECIMAL(3,2),
    
    -- Detected topics
    detected_topics TEXT[],
    
    -- LLM processing
    prompt_template_id VARCHAR(50),
    system_prompt TEXT,
    complete_prompt TEXT,
    
    -- LLM response
    llm_response TEXT NOT NULL,
    response_time_ms INTEGER,
    
    -- Backend actions
    backend_actions TEXT[],
    actions_executed JSONB,
    
    -- Metadata
    metadata JSONB,
    
    UNIQUE(conversation_id, turn_number)
);

CREATE INDEX idx_turns_conversation ON conversation_turns(conversation_id);
CREATE INDEX idx_turns_intent ON conversation_turns(user_intent);
CREATE INDEX idx_turns_timestamp ON conversation_turns(timestamp);


-- Topic Relationships table (Knowledge Graph)
CREATE TABLE IF NOT EXISTS topic_relationships (
    id SERIAL PRIMARY KEY,
    topic_a VARCHAR(50) NOT NULL,
    topic_b VARCHAR(50) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,  -- 'co-occurs', 'prerequisite', 'related', 'contradicts'
    strength DECIMAL(3,2) DEFAULT 0.5,  -- 0-1 strength of relationship
    evidence_count INTEGER DEFAULT 1,  -- How many times observed
    last_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(topic_a, topic_b, relationship_type)
);

CREATE INDEX idx_topic_rel_a ON topic_relationships(topic_a);
CREATE INDEX idx_topic_rel_b ON topic_relationships(topic_b);


-- Intent Transitions table (Conversation Flow Analysis)
CREATE TABLE IF NOT EXISTS intent_transitions (
    id SERIAL PRIMARY KEY,
    from_intent VARCHAR(50) NOT NULL,
    to_intent VARCHAR(50) NOT NULL,
    transition_count INTEGER DEFAULT 1,
    average_turns_between DECIMAL(4,2),
    success_rate DECIMAL(3,2),  -- Did user get answer?
    last_occurred TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(from_intent, to_intent)
);


-- Prompt Performance Metrics
CREATE TABLE IF NOT EXISTS prompt_performance (
    id SERIAL PRIMARY KEY,
    template_id VARCHAR(50) UNIQUE NOT NULL,
    times_used INTEGER DEFAULT 0,
    average_response_time_ms INTEGER,
    success_rate DECIMAL(3,2),  -- User satisfaction proxy
    feedback_score DECIMAL(3,2),
    last_used TIMESTAMP,
    
    -- Performance by topic
    performance_by_topic JSONB
);


-- User Feedback table
CREATE TABLE IF NOT EXISTS conversation_feedback (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(50) REFERENCES conversations(conversation_id),
    turn_number INTEGER,
    
    -- Feedback type
    feedback_type VARCHAR(20),  -- 'thumbs_up', 'thumbs_down', 'correction', 'suggestion'
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    
    -- Details
    feedback_text TEXT,
    corrected_response TEXT,  -- If user provides better answer
    
    -- Metadata
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Update topic relationships based on co-occurrence
CREATE OR REPLACE FUNCTION update_topic_relationships()
RETURNS TRIGGER AS $$
BEGIN
    -- For each pair of topics in this turn
    IF array_length(NEW.detected_topics, 1) > 1 THEN
        -- Insert or update co-occurrence relationships
        FOR i IN 1..array_length(NEW.detected_topics, 1) LOOP
            FOR j IN (i+1)..array_length(NEW.detected_topics, 1) LOOP
                INSERT INTO topic_relationships (topic_a, topic_b, relationship_type, evidence_count)
                VALUES (
                    NEW.detected_topics[i],
                    NEW.detected_topics[j],
                    'co-occurs',
                    1
                )
                ON CONFLICT (topic_a, topic_b, relationship_type)
                DO UPDATE SET
                    evidence_count = topic_relationships.evidence_count + 1,
                    strength = LEAST(1.0, topic_relationships.strength + 0.05),
                    last_observed = CURRENT_TIMESTAMP;
            END LOOP;
        END LOOP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update relationships
DROP TRIGGER IF EXISTS trg_update_topic_relationships ON conversation_turns;
CREATE TRIGGER trg_update_topic_relationships
    AFTER INSERT ON conversation_turns
    FOR EACH ROW
    EXECUTE FUNCTION update_topic_relationships();


-- Update intent transitions
CREATE OR REPLACE FUNCTION update_intent_transitions()
RETURNS TRIGGER AS $$
DECLARE
    prev_intent VARCHAR(50);
BEGIN
    -- Get previous turn's intent
    SELECT user_intent INTO prev_intent
    FROM conversation_turns
    WHERE conversation_id = NEW.conversation_id
      AND turn_number = NEW.turn_number - 1;
    
    -- If previous turn exists, record transition
    IF prev_intent IS NOT NULL THEN
        INSERT INTO intent_transitions (from_intent, to_intent, transition_count)
        VALUES (prev_intent, NEW.user_intent, 1)
        ON CONFLICT (from_intent, to_intent)
        DO UPDATE SET
            transition_count = intent_transitions.transition_count + 1,
            last_occurred = CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for transitions
DROP TRIGGER IF EXISTS trg_update_intent_transitions ON conversation_turns;
CREATE TRIGGER trg_update_intent_transitions
    AFTER INSERT ON conversation_turns
    FOR EACH ROW
    EXECUTE FUNCTION update_intent_transitions();


-- ============================================================================
-- VIEWS
-- ============================================================================

-- Most common conversation patterns
CREATE OR REPLACE VIEW v_common_conversation_patterns AS
SELECT
    from_intent,
    to_intent,
    transition_count,
    success_rate,
    ROUND((transition_count::DECIMAL / SUM(transition_count) OVER ()) * 100, 2) AS percentage
FROM intent_transitions
ORDER BY transition_count DESC
LIMIT 20;


-- Topic co-occurrence network
CREATE OR REPLACE VIEW v_topic_network AS
SELECT
    topic_a,
    topic_b,
    relationship_type,
    strength,
    evidence_count,
    CASE
        WHEN strength > 0.8 THEN 'STRONG'
        WHEN strength > 0.5 THEN 'MODERATE'
        ELSE 'WEAK'
    END AS relationship_strength_label
FROM topic_relationships
WHERE evidence_count >= 3  -- Only show if seen 3+ times
ORDER BY strength DESC, evidence_count DESC;


-- User engagement metrics
CREATE OR REPLACE VIEW v_user_engagement AS
SELECT
    user_id,
    COUNT(DISTINCT conversation_id) AS total_conversations,
    SUM(total_turns) AS total_turns,
    ROUND(AVG(total_turns), 1) AS avg_turns_per_conversation,
    ROUND(AVG(satisfaction_rating), 2) AS avg_satisfaction,
    MAX(session_start) AS last_active
FROM conversations
WHERE session_start >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY user_id
ORDER BY total_conversations DESC;


-- Popular topics by time period
CREATE OR REPLACE VIEW v_trending_topics AS
WITH topic_counts AS (
    SELECT
        unnest(detected_topics) AS topic,
        DATE_TRUNC('day', timestamp) AS day,
        COUNT(*) AS count
    FROM conversation_turns
    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY unnest(detected_topics), DATE_TRUNC('day', timestamp)
)
SELECT
    topic,
    SUM(count) AS total_mentions,
    ROUND(AVG(count), 1) AS avg_mentions_per_day,
    MAX(day) AS last_mentioned
FROM topic_counts
GROUP BY topic
ORDER BY total_mentions DESC;


-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Find most effective prompt templates
-- SELECT * FROM prompt_performance ORDER BY success_rate DESC, times_used DESC;

-- Analyze conversation flow from quote requests
-- SELECT * FROM v_common_conversation_patterns WHERE from_intent = 'quote_request';

-- Find related topics to "speeds_feeds"
-- SELECT * FROM v_topic_network WHERE topic_a = 'speeds_feeds' OR topic_b = 'speeds_feeds';

-- Get user conversation history
-- SELECT * FROM conversations WHERE user_id = 'USER-001' ORDER BY session_start DESC;

-- Trending topics this week
-- SELECT * FROM v_trending_topics;
