"""
Anti-Fragile Marketplace - Main Entry Point
Entry point for running the complete Anti-Fragile Marketplace system with
Survivor Ranking, Economic Audit, and Genetic Tracking.
"""

import sys
import argparse
from datetime import datetime
import logging
from pathlib import Path

# Add the main project path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .anti_fragile_marketplace import AntiFragileMarketplace
from .survivor_ranking import SurvivorRankingSystem
from .economic_auditor import EconomicAuditor
from .genetic_tracker import GeneticTracker
from ..app_factory import create_app
from ..repositories.telemetry_repository import TelemetryRepository
from ..services.dopamine_engine import DopamineEngine
from ..services.economics_engine import EconomicsEngine


def setup_logging():
    """Setup logging for the Anti-Fragile Marketplace"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('anti_fragile_marketplace.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main entry point for the Anti-Fragile Marketplace"""
    parser = argparse.ArgumentParser(description='Anti-Fragile Marketplace - G-Code Strategy Ranking System')
    parser.add_argument('--mode', choices=['simulation', 'evaluation', 'report'], 
                       default='simulation', help='Operation mode: simulation, evaluation, or report')
    parser.add_argument('--duration', type=float, default=1.0, 
                       help='Duration for simulation mode in hours (default: 1.0)')
    parser.add_argument('--material', type=str, help='Filter by material type')
    parser.add_argument('--operation', type=str, help='Filter by operation type')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Anti-Fragile Marketplace")
    logger.info(f"Arguments: mode={args.mode}, duration={args.duration}, "
                f"material={args.material}, operation={args.operation}")
    
    try:
        # Initialize the marketplace components
        logger.info("Initializing marketplace components...")
        
        # Create shared repository
        app = create_app()
        telemetry_repo = app.state.telemetry_repo
        
        # Initialize core systems
        survivor_ranking = SurvivorRankingSystem()
        economic_auditor = EconomicAuditor()
        genetic_tracker = GeneticTracker()
        
        # Create the marketplace
        marketplace = AntiFragileMarketplace()
        marketplace.ranking_system = survivor_ranking
        marketplace.economic_auditor = economic_auditor
        marketplace.genetic_tracker = genetic_tracker
        
        logger.info("Marketplace initialized successfully")
        
        if args.mode == 'simulation':
            logger.info(f"Running marketplace simulation for {args.duration} hours")
            results = marketplace.run_marketplace_simulation(duration_hours=args.duration)
            
            print(f"\nSIMULATION RESULTS:")
            print(f"  Duration: {args.duration} hours")
            print(f"  Strategies Evaluated: {results['strategies_evaluated']}")
            print(f"  Stress Tests Performed: {results['stress_tests_performed']}")
            print(f"  Survivor Badges Awarded: {results['survivor_badges_awarded']}")
            print(f"  Strategies Improved: {results['strategies_improved']}")
            print(f"  Economic Impact: ${results['economic_impact']:,.2f}")
            
            if results['top_performers']:
                print(f"\nTOP PERFORMERS:")
                for i, perf in enumerate(results['top_performers'][:5], 1):
                    print(f"  {i}. {perf['name']} - Score: {perf['survivor_score']:.3f}, "
                          f"Eco: {perf['economic_value']:.3f}, Badge: {perf['badge_level']}")
        
        elif args.mode == 'evaluation':
            logger.info("Running strategy evaluation")
            
            # Get top strategies based on filters
            top_strategies = marketplace.get_top_strategies(
                material=args.material,
                operation_type=args.operation,
                limit=10
            )
            
            print(f"\nTOP STRATEGIES{' FOR ' + args.material if args.material else ''}{' ' + args.operation if args.operation else ''}:")
            for i, strategy in enumerate(top_strategies, 1):
                print(f"  {i}. {strategy.strategy_name}")
                print(f"     Material: {strategy.material}, Operation: {strategy.operation_type}")
                print(f"     Survivor Score: {strategy.survivor_badge.survivor_score:.3f}")
                print(f"     Economic Value: {strategy.economic_value:.3f}")
                print(f"     Badge Level: {strategy.survivor_badge.badge_level}")
                print(f"     Tags: {', '.join(strategy.tags)}")
                print()
        
        elif args.mode == 'report':
            logger.info("Generating marketplace report")
            
            report_json = marketplace.generate_marketplace_report()
            print("\nMARKETPLACE REPORT:")
            print(report_json)
            
            # Save report to file
            report_filename = f"marketplace_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                f.write(report_json)
            print(f"\nReport saved to {report_filename}")
        
        logger.info("Anti-Fragile Marketplace completed successfully")
        
    except Exception as e:
        logger.error(f"Anti-Fragile Marketplace failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()