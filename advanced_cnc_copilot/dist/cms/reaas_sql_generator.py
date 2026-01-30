"""
SQL Generator for REaaS Synthetic Data
Converts simulated reverse engineering data to SQL templates
"""

import json
from typing import List
from cms.reverse_engineering_simulator import (
    SyntheticProject, TelemetryPoint, ReverseEngineeringSimulator
)


class REaaSSQLGenerator:
    """
    Generates SQL compatible with ERP schema
    Includes project, job, and telemetry tables
    """
    
    def __init__(self):
        self.project_counter = 1
        self.job_counter = 1
        self.telemetry_counter = 1
    
    def generate_complete_sql(self, 
                             project: SyntheticProject,
                             telemetry: List[TelemetryPoint],
                             sample_telemetry: int = 100) -> str:
        """
        Generate complete SQL for REaaS project
        
        Args:
            project: Project specification
            telemetry: Telemetry data
            sample_telemetry: How many telemetry points to include
        
        Returns:
            Complete SQL script
        """
        sql = []
        
        # Header
        sql.append(f"-- Reverse Engineering Project: {project.name}")
        sql.append(f"-- Generated: {project.created_at}")
        sql.append(f"-- Scenario: {project.failure_scenario}\n")
        
        # Project insert
        sql.append(self._generate_project_sql(project))
        
        # Job insert
        sql.append(self._generate_job_sql())
        
        # Telemetry insert (sampled)
        step = max(1, len(telemetry) // sample_telemetry)
        sampled = telemetry[::step][:sample_telemetry]
        
        sql.append(self._generate_telemetry_sql(sampled))
        
        self.project_counter += 1
        
        return "\n".join(sql)
    
    def _generate_project_sql(self, project: SyntheticProject) -> str:
        """Generate erp_project INSERT"""
        
        # Create LLM suggestions (would come from actual LLM)
        llm_suggestions = {
            "material_recommendation": project.material,
            "complexity_analysis": f"Complexity score {project.complexity:.2f} - "
                                  f"{'High' if project.complexity > 0.7 else 'Medium' if project.complexity > 0.4 else 'Low'}",
            "tooling_suggestions": [
                "Carbide end mill for roughing",
                "CBN insert for finishing" if "Inconel" in project.material else "Coated carbide for finishing"
            ],
            "risk_factors": [
                "Material hardness requires careful tool selection",
                f"Wear deviation of {project.dimensions['max_deviation']:.3f}mm indicates significant usage"
            ]
        }
        
        sql = f"""
-- Project: {project.name}
INSERT INTO erp_project (
    name,
    part_number,
    gcode,
    created_at,
    material,
    complexity_score,
    estimated_cycle_time,
    dimensions,
    llm_suggestions,
    success,
    vendor_name,
    part_type,
    failure_scenario
) VALUES (
    '{project.name}',
    '{project.part_number}',
    '(G-Code generated from reverse-engineered CAD model)',
    '{project.created_at}',
    '{project.material}',
    {project.complexity},
    {project.estimated_cycle_time},
    '{json.dumps(project.dimensions)}',
    '{json.dumps(llm_suggestions)}',
    1,
    '{project.vendor}',
    '{project.part_type}',
    '{project.failure_scenario}'
);
"""
        return sql
    
    def _generate_job_sql(self) -> str:
        """Generate erp_job INSERT"""
        sql = f"""
-- Job for project {self.project_counter}
INSERT INTO erp_job (
    job_id,
    quantity,
    completed_quantity,
    priority,
    status,
    project_id,
    machine_id,
    started_at
) VALUES (
    'JOB-{self.job_counter:06d}',
    1,
    1,
    1,
    'COMPLETED',
    {self.project_counter},
    1,
    datetime('now')
);
"""
        self.job_counter += 1
        return sql
    
    def _generate_telemetry_sql(self, telemetry: List[TelemetryPoint]) -> str:
        """Generate erp_telemetry INSERTs"""
        sql = ["\n-- Telemetry Data (Biochemical + Physical)"]
        
        for point in telemetry:
            sql.append(f"""
INSERT INTO erp_telemetry (
    timestamp,
    rpm,
    load,
    vibration_x,
    vibration_y,
    vibration_z,
    temperature_c,
    tool_health,
    cortisol,
    dopamine,
    serotonin,
    adrenaline,
    signal,
    anomaly_detected,
    active_tool,
    machine_id,
    job_id
) VALUES (
    '{point.timestamp}',
    {point.rpm},
    {point.spindle_load},
    {point.vibration_x},
    {point.vibration_y},
    {point.vibration_z},
    {point.temperature_c},
    {point.tool_health},
    {point.cortisol},
    {point.dopamine},
    {point.serotonin},
    {point.adrenaline},
    '{point.signal}',
    {1 if point.anomaly_detected else 0},
    '{point.tool_id}',
    {point.machine_id},
    'JOB-{self.job_counter-1:06d}'
);""")
        
        return "\n".join(sql)
    
    def generate_batch_sql(self, num_projects: int = 10, output_file: str = "reaas_synthetic_data.sql") -> str:
        """
        Generate multiple REaaS projects
        
        Args:
            num_projects: Number of projects to generate
            output_file: Output file name
        
        Returns:
            Path to generated file
        """
        simulator = ReverseEngineeringSimulator()
        
        all_sql = []
        all_sql.append("-- REaaS Synthetic Data Generation")
        all_sql.append(f"-- Total Projects: {num_projects}\n")
        all_sql.append("BEGIN TRANSACTION;\n")
        
        for i in range(num_projects):
            print(f"Generating project {i+1}/{num_projects}...")
            
            # Generate project
            project = simulator.generate_project_context()
            
            # Simulate production
            telemetry = simulator.simulate_production_run(
                project,
                duration_seconds=project.estimated_cycle_time * 60,
                sample_rate_hz=1  # 1 Hz for SQL
            )
            
            # Generate SQL
            sql = self.generate_complete_sql(project, telemetry, sample_telemetry=50)
            all_sql.append(sql)
            all_sql.append("\n")
        
        all_sql.append("COMMIT;")
        
        # Write to file
        full_sql = "\n".join(all_sql)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_sql)
        
        print(f"✅ Generated {output_file}")
        print(f"   Projects: {num_projects}")
        print(f"   Telemetry points per project: ~50")
        
        return output_file


if __name__ == "__main__":
    generator = REaaSSQLGenerator()
    
    # Generate batch
    generator.generate_batch_sql(num_projects=5, output_file="reaas_synthetic_data.sql")
