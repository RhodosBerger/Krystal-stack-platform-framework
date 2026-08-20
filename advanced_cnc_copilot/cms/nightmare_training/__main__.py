"""
Nightmare Training - Main Entry Point
Entry point for running nightmare training sessions
"""

import sys
import argparse
from datetime import datetime
import logging
from pathlib import Path

# Add the main project path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .orchestrator import NightmareTrainingOrchestrator
from ..app_factory import create_app
from ..models import get_session_local, get_database_url
from ..repositories.telemetry_repository import TelemetryRepository
from ..services.shadow_council import ShadowCouncil, CreatorAgent, AuditorAgent, DecisionPolicy, AccountantAgent
from ..services.dopamine_engine import DopamineEngine
from ..services.economics_engine import EconomicsEngine
from ..models import create_database_engine, create_tables


def setup_logging():
    """Setup logging for nightmare training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nightmare_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main entry point for nightmare training"""
    parser = argparse.ArgumentParser(description='Nightmare Training - Offline Learning for CNC Copilot')
    parser.add_argument('--machine-id', type=int, help='Specific machine ID to train on')
    parser.add_argument('--duration', type=float, default=1.0, 
                       help='Duration of historical data to replay in hours (default: 1.0)')
    parser.add_argument('--probability', type=float, default=0.7,
                       help='Probability of injecting failures (0.0 to 1.0, default: 0.7)')
    parser.add_argument('--idle-check', action='store_true',
                       help='Check if machine is idle before training')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Nightmare Training session")
    logger.info(f"Arguments: machine_id={args.machine_id}, duration={args.duration}, "
                f"probability={args.probability}")
    
    try:
        # Create database engine and session
        engine = create_database_engine()
        create_tables(engine)
        db_session = get_session_local(engine)()
        
        # Initialize required components manually
        telemetry_repo = TelemetryRepository(db_session)
        dopamine_engine = DopamineEngine(repository=telemetry_repo)
        economics_engine = EconomicsEngine(repository=telemetry_repo)
        
        # Initialize Shadow Council components
        decision_policy = DecisionPolicy()
        creator_agent = CreatorAgent(repository=telemetry_repo)
        auditor_agent = AuditorAgent(decision_policy=decision_policy)
        accountant_agent = AccountantAgent(economics_engine=economics_engine)
        
        # Initialize Shadow Council governance
        shadow_council = ShadowCouncil(
            creator=creator_agent,
            auditor=auditor_agent,
            decision_policy=decision_policy
        )
        shadow_council.set_accountant(accountant_agent)
        
        # Create orchestrator
        orchestrator = NightmareTrainingOrchestrator(shadow_council, telemetry_repo)
        
        # Determine which machines to train
        if args.machine_id:
            machine_ids = [args.machine_id]
        else:
            # For now, default to machine ID 1 - in practice, this would come from configuration
            machine_ids = [1]
        
        # Run nightmare training for each machine
        for machine_id in machine_ids:
            logger.info(f"Starting nightmare training for machine {machine_id}")
            
            # Prepare schedule configuration
            schedule_config = {
                'duration_hours': args.duration,
                'failure_probability': args.probability,
                'force_during_operation': not args.idle_check
            }
            
            # Run a single session or schedule multiple
            if args.machine_id:
                # Run single session
                result = orchestrator.run_nightmare_training_session(
                    machine_id=machine_id,
                    duration_hours=args.duration,
                    failure_probability=args.probability
                )
                logger.info(f"Completed session for machine {machine_id}: {result['status']}")
                logger.info(f"Summary: Kill Switch Triggers={result['summary']['total_kill_switch_triggers']}, "
                           f"Preemptive Responses={result['summary']['total_preemptive_responses']}, "
                           f"Missed Failures={result['summary']['total_missed_failures']}")
            else:
                # Schedule sessions
                results = orchestrator.schedule_nightmare_training(machine_ids, schedule_config)
                logger.info(f"Scheduled {len(results['scheduled_sessions'])} sessions, "
                           f"{len(results['failed_sessions'])} failed")
        
        logger.info("Nightmare Training completed successfully")
        
    except Exception as e:
        logger.error(f"Nightmare Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()