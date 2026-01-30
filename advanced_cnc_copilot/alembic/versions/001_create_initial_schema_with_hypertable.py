"""Create initial schema with hypertable for telemetry

Revision ID: 001
Revises: 
Create Date: 2026-01-26 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create the machines table
    op.create_table(
        'machines',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('serial_number', sa.String(), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_seen', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('serial_number')
    )
    op.create_index(op.f('ix_machines_id'), 'machines', ['id'], unique=False)
    op.create_index(op.f('ix_machines_name'), 'machines', ['name'], unique=False)

    # Create the projects table
    op.create_table(
        'projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('estimated_duration_hours', sa.Float(), nullable=True),
        sa.Column('actual_duration_hours', sa.Float(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_projects_id'), 'projects', ['id'], unique=False)

    # Create the telemetry table (before converting to hypertable)
    op.create_table(
        'telemetry',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('machine_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('spindle_load', sa.Float(), nullable=False),
        sa.Column('vibration_x', sa.Float(), nullable=False),
        sa.Column('dopamine_score', sa.Float(), nullable=True),
        sa.Column('cortisol_level', sa.Float(), nullable=True),
        sa.Column('spindle_rpm', sa.Float(), nullable=True),
        sa.Column('feed_rate', sa.Float(), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('axis_position_x', sa.Float(), nullable=True),
        sa.Column('axis_position_y', sa.Float(), nullable=True),
        sa.Column('axis_position_z', sa.Float(), nullable=True),
        sa.Column('tool_offset_x', sa.Float(), nullable=True),
        sa.Column('tool_offset_y', sa.Float(), nullable=True),
        sa.Column('tool_offset_z', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_telemetry_id'), 'telemetry', ['id'], unique=False)
    op.create_index(op.f('ix_telemetry_timestamp'), 'telemetry', ['timestamp'], unique=False)
    op.create_index(op.f('ix_telemetry_machine_id'), 'telemetry', ['machine_id'], unique=False)

    # Convert the telemetry table to a TimescaleDB hypertable
    # This is done with raw SQL since Alembic doesn't have built-in support for hypertables
    op.execute("""
        SELECT create_hypertable('telemetry', 'timestamp', 
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE);
    """)

    # Create indexes optimized for time-series queries
    op.create_index('idx_telemetry_machine_time', 'telemetry', ['machine_id', 'timestamp'])
    op.create_index('idx_telemetry_cortisol_time', 'telemetry', ['cortisol_level', 'timestamp'])
    op.create_index('idx_telemetry_dopamine_time', 'telemetry', ['dopamine_score', 'timestamp'])


def downgrade():
    # Drop the hypertable (which will also drop the underlying table)
    op.execute("DROP TABLE IF EXISTS telemetry;")
    
    # Drop other tables
    op.drop_index(op.f('ix_projects_id'), table_name='projects')
    op.drop_table('projects')
    
    op.drop_index(op.f('ix_machines_name'), table_name='machines')
    op.drop_index(op.f('ix_machines_id'), table_name='machines')
    op.drop_table('machines')