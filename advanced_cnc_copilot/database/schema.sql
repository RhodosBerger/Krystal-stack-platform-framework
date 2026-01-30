-- ============================================================================
-- Manufacturing Intelligence Platform - PostgreSQL Schema
-- ============================================================================
-- 
-- This script creates the complete database schema for the manufacturing
-- intelligence platform with 9 core tables.
--
-- TABLES:
-- 1. vendors - Supplier information
-- 2. materials - Material properties library
-- 3. producers - Manufacturing capability database
-- 4. parts - Part catalog
-- 5. projects - Manufacturing projects/orders
-- 6. jobs - Production jobs within projects
-- 7. operations - Individual machining operations
-- 8. telemetry - Real-time sensor data (PARTITIONED)
-- 9. quality_inspections - QC records
--
-- Run this script with:
-- psql -U postgres -d manufacturing_db -f schema.sql
-- ============================================================================

-- Create database (if needed)
-- CREATE DATABASE manufacturing_db;

-- Connect to database
\c manufacturing_db

-- Enable TimescaleDB extension for time-series data (optional but recommended)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- MASTER DATA TABLES
-- ============================================================================

-- Vendors table
CREATE TABLE IF NOT EXISTS vendors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    contact_info JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_vendors_name ON vendors(name);

-- Materials table
CREATE TABLE IF NOT EXISTS materials (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50),
    properties JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_materials_name ON materials(name);
CREATE INDEX idx_materials_type ON materials(type);

-- Producers table
CREATE TABLE IF NOT EXISTS producers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100),
    location VARCHAR(255),
    capacity JSONB,
    capabilities JSONB,
    effectiveness_score DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_producers_name ON producers(name);
CREATE INDEX idx_producers_effectiveness ON producers(effectiveness_score);

-- Parts table
CREATE TABLE IF NOT EXISTS parts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    part_number VARCHAR(100) NOT NULL UNIQUE,
    vendor_id INTEGER REFERENCES vendors(id) ON DELETE SET NULL,
    material_id INTEGER REFERENCES materials(id) ON DELETE SET NULL,
    dimensions JSONB,
    weight_kg DOUBLE PRECISION,
    complexity_score DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_parts_part_number ON parts(part_number);
CREATE INDEX idx_parts_vendor ON parts(vendor_id);
CREATE INDEX idx_parts_material ON parts(material_id);

-- ============================================================================
-- CORE OPERATIONAL TABLES
-- ============================================================================

-- Project status enum
CREATE TYPE project_status AS ENUM ('pending', 'in_progress', 'completed', 'cancelled');

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    due_date TIMESTAMP,
    status project_status DEFAULT 'pending',
    llm_suggestions JSONB,
    estimated_cost DOUBLE PRECISION,
    actual_cost DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_dates ON projects(start_date, end_date);

-- Job status enum
CREATE TYPE job_status AS ENUM ('queued', 'running', 'paused', 'completed', 'failed');

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(50) NOT NULL UNIQUE,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    part_id INTEGER NOT NULL REFERENCES parts(id) ON DELETE RESTRICT,
    producer_id INTEGER REFERENCES producers(id) ON DELETE SET NULL,
    quantity INTEGER NOT NULL,
    completed_quantity INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 1,
    due_date TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status job_status DEFAULT 'queued',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_jobs_project ON jobs(project_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_job_id ON jobs(job_id);

-- Operation status enum
CREATE TYPE operation_status AS ENUM ('queued', 'running', 'completed', 'failed');

-- Operations table
CREATE TABLE IF NOT EXISTS operations (
    id SERIAL PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    sequence_order INTEGER,
    operation_type VARCHAR(100),
    estimated_time INTERVAL,
    actual_time INTERVAL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    parameters JSONB,
    status operation_status DEFAULT 'queued',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_operations_job ON operations(job_id);
CREATE INDEX idx_operations_status ON operations(status);
CREATE INDEX idx_operations_sequence ON operations(job_id, sequence_order);

-- ============================================================================
-- TIME-SERIES & QUALITY TABLES
-- ============================================================================

-- Telemetry table (PARTITIONED by timestamp)
-- This table will have MILLIONS of rows, so partitioning is critical
CREATE TABLE IF NOT EXISTS telemetry (
    id BIGSERIAL,
    operation_id INTEGER NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    sensor_id VARCHAR(100),
    value DOUBLE PRECISION,
    unit VARCHAR(50),
    metadata JSONB,
    cortisol DOUBLE PRECISION,
    dopamine DOUBLE PRECISION,
    serotonin DOUBLE PRECISION,
    adrenaline DOUBLE PRECISION,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for current month (example - create more as needed)
CREATE TABLE IF NOT EXISTS telemetry_2026_01 PARTITION OF telemetry
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE IF NOT EXISTS telemetry_2026_02 PARTITION OF telemetry
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- Indexes on telemetry
CREATE INDEX idx_telemetry_operation ON telemetry(operation_id);
CREATE INDEX idx_telemetry_timestamp ON telemetry(timestamp DESC);
CREATE INDEX idx_telemetry_sensor ON telemetry(sensor_id);

-- Quality result enum
CREATE TYPE quality_result AS ENUM ('pass', 'fail', 'conditional');

-- Quality inspections table
CREATE TABLE IF NOT EXISTS quality_inspections (
    id SERIAL PRIMARY KEY,
    operation_id INTEGER NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
    inspector_id VARCHAR(100),
    result quality_result NOT NULL,
    measurements JSONB,
    defects JSONB,
    comments TEXT,
    inspection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_quality_operation ON quality_inspections(operation_id);
CREATE INDEX idx_quality_result ON quality_inspections(result);
CREATE INDEX idx_quality_date ON quality_inspections(inspection_date DESC);

-- ============================================================================
-- INITIAL DATA - Materials
-- ============================================================================

INSERT INTO materials (name, type, properties) VALUES
('Aluminum 6061', 'Metal', '{
    "density": 2.7,
    "tensile_strength": 310,
    "yield_strength": 276,
    "hardness_hb": 95,
    "elastic_modulus": 69000,
    "thermal_conductivity": 167,
    "melting_point": 582,
    "cost_factor": 1.0
}'::jsonb),
('Steel 4140', 'Metal', '{
    "density": 7.85,
    "tensile_strength": 655,
    "yield_strength": 415,
    "hardness_hb": 250,
    "elastic_modulus": 210000,
    "thermal_conductivity": 42.6,
    "melting_point": 1416,
    "cost_factor": 0.6
}'::jsonb),
('Titanium Ti-6Al-4V', 'Metal', '{
    "density": 4.43,
    "tensile_strength": 900,
    "yield_strength": 830,
    "hardness_hb": 330,
    "elastic_modulus": 114000,
    "thermal_conductivity": 7.2,
    "melting_point": 1604,
    "cost_factor": 8.0
}'::jsonb),
('Stainless Steel 316', 'Metal', '{
    "density": 8.0,
    "tensile_strength": 515,
    "yield_strength": 205,
    "hardness_hb": 217,
    "elastic_modulus": 193000,
    "thermal_conductivity": 16.3,
    "melting_point": 1375,
    "cost_factor": 2.5
}'::jsonb)
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- INITIAL DATA - Producers
-- ============================================================================

INSERT INTO producers (name, type, location, capacity, capabilities, effectiveness_score) VALUES
('Precision Manufacturing Inc.', 'external', 'Detroit, MI', '{
    "max_parts_per_day": 500,
    "machines": 12,
    "employees": 45
}'::jsonb, '{
    "materials": ["Aluminum", "Steel", "Stainless"],
    "processes": ["3-axis CNC", "4-axis CNC", "Turning"],
    "min_tolerance": 0.01,
    "max_complexity": 0.8
}'::jsonb, 0.85),
('HighTech Aerospace Solutions', 'external', 'Los Angeles, CA', '{
    "max_parts_per_day": 200,
    "machines": 8,
    "employees": 30
}'::jsonb, '{
    "materials": ["Titanium", "Aluminum", "Inconel"],
    "processes": ["5-axis CNC", "EDM", "Inspection"],
    "min_tolerance": 0.005,
    "max_complexity": 1.0
}'::jsonb, 0.92),
('BudgetParts LLC', 'external', 'Cleveland, OH', '{
    "max_parts_per_day": 1000,
    "machines": 20,
    "employees": 60
}'::jsonb, '{
    "materials": ["Aluminum", "Steel", "Plastic"],
    "processes": ["3-axis CNC", "Turning"],
    "min_tolerance": 0.05,
    "max_complexity": 0.5
}'::jsonb, 0.68)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- VIEWS - Useful pre-built queries
-- ============================================================================

-- View: Project progress summary
CREATE OR REPLACE VIEW project_progress AS
SELECT 
    p.id,
    p.name,
    p.status,
    COUNT(j.id) as total_jobs,
    COUNT(j.id) FILTER (WHERE j.status = 'completed') as completed_jobs,
    ROUND(100.0 * COUNT(j.id) FILTER (WHERE j.status = 'completed') / COUNT(j.id), 2) as completion_percentage,
    p.estimated_cost,
    p.actual_cost
FROM projects p
LEFT JOIN jobs j ON p.id = j.project_id
GROUP BY p.id;

-- View: Real-time operation status
CREATE OR REPLACE VIEW operation_status_live AS
SELECT 
    o.id,
    o.name as operation_name,
    j.job_id,
    p.name as part_name,
    o.status,
    o.started_at,
    o.estimated_time,
    EXTRACT(EPOCH FROM (NOW() - o.started_at)) as elapsed_seconds,
    EXTRACT(EPOCH FROM o.estimated_time) as estimated_seconds
FROM operations o
JOIN jobs j ON o.job_id = j.id
JOIN parts p ON j.part_id = p.id
WHERE o.status = 'running';

-- ============================================================================
-- FUNCTIONS - Stored procedures
-- ============================================================================

-- Function: Get latest telemetry for operation
CREATE OR REPLACE FUNCTION get_latest_telemetry(op_id INTEGER, sensor_name VARCHAR)
RETURNS TABLE (
    timestamp TIMESTAMP,
    value DOUBLE PRECISION,
    unit VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT t.timestamp, t.value, t.unit
    FROM telemetry t
    WHERE t.operation_id = op_id
      AND t.sensor_id = sensor_name
    ORDER BY t.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS - Automatic updates
-- ============================================================================

-- Trigger: Update projects.updated_at on any change
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vendors_updated_at BEFORE UPDATE ON vendors
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PERMISSIONS - Security (adjust as needed)
-- ============================================================================

-- Create read-only user for dashboards
-- CREATE USER dashboard_user WITH PASSWORD 'your_secure_password';
-- GRANT CONNECT ON DATABASE manufacturing_db TO dashboard_user;
-- GRANT USAGE ON SCHEMA public TO dashboard_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_user;

-- ============================================================================
-- COMPLETE
-- ============================================================================

\echo '✅ Database schema created successfully!'
\echo ''
\echo '📊 Tables created:'
\echo '  1. vendors'
\echo '  2. materials (with 4 initial materials)'
\echo '  3. producers (with 3 initial producers)'
\echo '  4. parts'
\echo '  5. projects'
\echo '  6. jobs'
\echo '  7. operations'
\echo '  8. telemetry (partitioned by month)'
\echo '  9. quality_inspections'
\echo ''
\echo '📋 Views created:'
\echo '  - project_progress'
\echo '  - operation_status_live'
\echo ''
\echo '⚡ Functions created:'
\echo '  - get_latest_telemetry'
\echo ''
\echo '🔐 Next steps:'
\echo '  1. Update DATABASE_URL in .env file'
\echo '  2. Create partitions for future months'
\echo '  3. Set up automated backups'
\echo '  4. Configure monitoring (pg_stat_statements)'
