"""
Anti-Fragile Marketplace - Main Orchestration Engine
Combines Survivor Ranking, Economic Audit, and Genetic Tracking into a unified marketplace
that ranks G-Code strategies by resilience rather than speed.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json
from dataclasses import dataclass
import threading

from .survivor_ranking import SurvivorRankingSystem, SurvivorBadge
from .economic_auditor import EconomicAuditor
from .genetic_tracker import GeneticTracker, GeneticLineage, MutationType


@dataclass
class MarketplaceListing:
    """A single listing in the Anti-Fragile Marketplace"""
    strategy_id: str
    strategy_name: str
    material: str
    operation_type: str
    survivor_badge: SurvivorBadge
    genetic_lineage: GeneticLineage
    economic_value: float  # Economic benefit rating
    timestamp: datetime
    author: str
    tags: List[str]
    description: str = ""
    version: str = "1.0"


class AntiFragileMarketplace:
    """
    The Anti-Fragile Marketplace - Where G-Code strategies compete based on resilience,
    not speed. Strategies earn "Survivor Badges" and "Genetic Lineage Rankings" that
    determine their market position based on their ability to survive chaotic conditions.
    """
    
    def __init__(self):
        self.ranking_system = SurvivorRankingSystem()
        self.economic_auditor = EconomicAuditor()
        self.genetic_tracker = GeneticTracker()
        self.listings = {}  # strategy_id -> MarketplaceListing
        self.marketplace_lock = threading.Lock()
        self.transaction_history = []
        
    def submit_strategy(self, strategy_id: str, strategy_name: str, 
                       material: str, operation_type: str, parameters: Dict[str, Any],
                       author: str, description: str = "") -> MarketplaceListing:
        """
        Submit a G-Code strategy to the marketplace for evaluation and ranking.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_name: Human-readable name
            material: Material the strategy is designed for
            operation_type: Type of operation (face_mill, drill, etc.)
            parameters: Strategy parameters
            author: Who submitted the strategy
            description: Optional description of the strategy
            
        Returns:
            MarketplaceListing with the evaluated strategy
        """
        with self.marketplace_lock:
            # First, evaluate the strategy's resilience using the Survivor Ranking System
            survivor_badge = self.ranking_system.evaluate_strategy_resilience(
                strategy_id, strategy_name, material, operation_type
            )
            
            # Create initial genetic lineage
            genetic_lineage = self.genetic_tracker.register_initial_strategy(
                strategy_id, material, operation_type, parameters
            )
            
            # Calculate economic value based on resilience and efficiency
            economic_value = self._calculate_economic_value(survivor_badge, parameters)
            
            # Create marketplace listing
            listing = MarketplaceListing(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                material=material,
                operation_type=operation_type,
                survivor_badge=survivor_badge,
                genetic_lineage=genetic_lineage,
                economic_value=economic_value,
                timestamp=datetime.utcnow(),
                author=author,
                tags=self._generate_strategy_tags(survivor_badge, material, operation_type),
                description=description,
                version="1.0"
            )
            
            # Store the listing
            self.listings[strategy_id] = listing
            
            # Record transaction
            self.transaction_history.append({
                'action': 'strategy_submitted',
                'strategy_id': strategy_id,
                'author': author,
                'timestamp': datetime.utcnow().isoformat(),
                'survivor_score': survivor_badge.survivor_score,
                'economic_value': economic_value
            })
            
            print(f"[MARKETPLACE] Strategy submitted: {strategy_name} "
                  f"(ID: {strategy_id}) by {author}")
            print(f"  Survivor Score: {survivor_badge.survivor_score:.3f}")
            print(f"  Economic Value: {economic_value:.3f}")
            print(f"  Badge Level: {survivor_badge.badge_level}")
            
            return listing
    
    def _calculate_economic_value(self, survivor_badge: SurvivorBadge, 
                                 parameters: Dict[str, Any]) -> float:
        """
        Calculate the economic value of a strategy based on its resilience and efficiency.
        
        Args:
            survivor_badge: The resilience rating of the strategy
            parameters: Strategy parameters that affect efficiency
            
        Returns:
            Economic value score (0.0 to 1.0)
        """
        # Base economic value on survivor score (resilience)
        base_value = survivor_badge.survivor_score
        
        # Adjust for efficiency parameters (higher feed rates, RPMs have potential for more economic value)
        efficiency_boost = 0.0
        if 'feed_rate' in parameters:
            # Normalize feed rate to 0-1 scale (assuming max reasonable feed rate is 5000)
            normalized_feed = min(1.0, parameters['feed_rate'] / 5000.0)
            efficiency_boost += normalized_feed * 0.1
        
        if 'rpm' in parameters:
            # Normalize RPM to 0-1 scale (assuming max reasonable RPM is 12000)
            normalized_rpm = min(1.0, parameters['rpm'] / 12000.0)
            efficiency_boost += normalized_rpm * 0.1
        
        # Adjust for complexity factor
        complexity_value = (survivor_badge.complexity_factor - 1.0) * 0.2
        
        # Combine all factors
        economic_value = min(1.0, base_value + efficiency_boost + complexity_value)
        
        return max(0.0, economic_value)  # Ensure non-negative value
    
    def _generate_strategy_tags(self, survivor_badge: SurvivorBadge, 
                               material: str, operation_type: str) -> List[str]:
        """
        Generate descriptive tags for a strategy based on its characteristics.
        
        Args:
            survivor_badge: The resilience rating of the strategy
            material: Material the strategy is for
            operation_type: Operation type
            
        Returns:
            List of relevant tags
        """
        tags = [material.lower().replace('-', '_'), operation_type.lower().replace(' ', '_')]
        
        # Add tags based on survivor score
        if survivor_badge.survivor_score >= 0.9:
            tags.append('ultra_resilient')
            tags.append('battle_tested')
        elif survivor_badge.survivor_score >= 0.7:
            tags.append('highly_resilient')
        elif survivor_badge.survivor_score >= 0.5:
            tags.append('moderately_resilient')
        
        # Add tags based on badge level
        badge_level_lower = survivor_badge.badge_level.lower()
        tags.append(badge_level_lower)
        
        # Add environmental resilience tags
        for env_type, resilience in survivor_badge.environmental_resilience.items():
            if resilience >= 0.9:
                tags.append(f'excel_{env_type}_resistant')
            elif resilience >= 0.7:
                tags.append(f'{env_type}_resistant')
        
        return list(set(tags))  # Remove duplicates
    
    def get_top_strategies(self, material: Optional[str] = None, operation_type: Optional[str] = None,
                          min_survivor_score: float = 0.0, 
                          limit: int = 10) -> List[MarketplaceListing]:
        """
        Get top-ranked strategies based on survivor score and economic value.
        
        Args:
            material: Filter by material (optional)
            operation_type: Filter by operation type (optional)
            min_survivor_score: Minimum survivor score threshold
            limit: Maximum number of results to return
            
        Returns:
            List of top strategies sorted by combined score
        """
        # Filter listings based on criteria
        candidates = []
        
        for listing in self.listings.values():
            # Apply filters
            if material and listing.material.lower() != material.lower():
                continue
            if operation_type and listing.operation_type.lower() != operation_type.lower():
                continue
            if listing.survivor_badge.survivor_score < min_survivor_score:
                continue
            
            candidates.append(listing)
        
        # Sort by combined score (survivor score + economic value)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (x.survivor_badge.survivor_score + x.economic_value) / 2,
            reverse=True
        )
        
        return sorted_candidates[:limit]
    
    def get_marketplace_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the marketplace.
        
        Returns:
            Dictionary with marketplace statistics
        """
        total_strategies = len(self.listings)
        
        # Calculate statistics by material
        material_stats = {}
        operation_stats = {}
        badge_level_counts = {}
        score_ranges = {
            'diamond': 0,  # 0.95+
            'platinum': 0,  # 0.85-0.94
            'gold': 0,     # 0.70-0.84
            'silver': 0,   # 0.50-0.69
            'bronze': 0    # < 0.50
        }
        
        for listing in self.listings.values():
            # Material stats
            material = listing.material
            if material not in material_stats:
                material_stats[material] = {
                    'count': 0,
                    'avg_survivor_score': 0.0,
                    'avg_economic_value': 0.0,
                    'total_strategies': 0
                }
            mat_stat = material_stats[material]
            mat_stat['count'] += 1
            mat_stat['avg_survivor_score'] += listing.survivor_badge.survivor_score
            mat_stat['avg_economic_value'] += listing.economic_value
            mat_stat['total_strategies'] += 1
            
            # Operation stats
            operation = listing.operation_type
            if operation not in operation_stats:
                operation_stats[operation] = {
                    'count': 0,
                    'avg_survivor_score': 0.0,
                    'avg_economic_value': 0.0
                }
            op_stat = operation_stats[operation]
            op_stat['count'] += 1
            op_stat['avg_survivor_score'] += listing.survivor_badge.survivor_score
            op_stat['avg_economic_value'] += listing.economic_value
            
            # Badge level counts
            badge_level = listing.survivor_badge.badge_level.lower()
            if badge_level not in badge_level_counts:
                badge_level_counts[badge_level] = 0
            badge_level_counts[badge_level] += 1
            
            # Score ranges
            score = listing.survivor_badge.survivor_score
            if score >= 0.95:
                score_ranges['diamond'] += 1
            elif score >= 0.85:
                score_ranges['platinum'] += 1
            elif score >= 0.70:
                score_ranges['gold'] += 1
            elif score >= 0.50:
                score_ranges['silver'] += 1
            else:
                score_ranges['bronze'] += 1
        
        # Calculate averages
        for mat, stats in material_stats.items():
            if stats['total_strategies'] > 0:
                stats['avg_survivor_score'] /= stats['total_strategies']
                stats['avg_economic_value'] /= stats['total_strategies']
        
        for op, stats in operation_stats.items():
            if stats['count'] > 0:
                stats['avg_survivor_score'] /= stats['count']
                stats['avg_economic_value'] /= stats['count']
        
        # Get marketplace-wide metrics
        all_scores = [l.survivor_badge.survivor_score for l in self.listings.values()]
        all_economic_values = [l.economic_value for l in self.listings.values()]
        
        marketplace_metrics = {
            'total_strategies': total_strategies,
            'average_survivor_score': sum(all_scores) / len(all_scores) if all_scores else 0.0,
            'average_economic_value': sum(all_economic_values) / len(all_economic_values) if all_economic_values else 0.0,
            'highest_survivor_score': max(all_scores) if all_scores else 0.0,
            'lowest_survivor_score': min(all_scores) if all_scores else 0.0,
            'strategy_growth_rate': self._calculate_growth_rate(),
            'top_materials': sorted(material_stats.items(), 
                                   key=lambda x: x[1]['count'], reverse=True)[:5],
            'top_operations': sorted(operation_stats.items(), 
                                    key=lambda x: x[1]['count'], reverse=True)[:5],
            'badge_distribution': badge_level_counts,
            'score_distribution': score_ranges,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return marketplace_metrics
    
    def _calculate_growth_rate(self) -> float:
        """Calculate the growth rate of strategies in the marketplace."""
        if len(self.transaction_history) < 2:
            return 0.0
        
        # Calculate strategies added in the last week
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        recent_additions = 0
        for transaction in self.transaction_history:
            if transaction['action'] == 'strategy_submitted':
                trans_time = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
                if trans_time >= week_ago:
                    recent_additions += 1
        
        return recent_additions / 7.0  # Average per day
    
    def get_strategy_genealogy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the complete genealogy and evolution history of a strategy.
        
        Args:
            strategy_id: ID of the strategy to get genealogy for
            
        Returns:
            Dictionary with genealogy information
        """
        # Get the listing
        listing = self.listings.get(strategy_id)
        if not listing:
            return {"error": f"Strategy {strategy_id} not found"}
        
        # Get the genetic lineage
        lineage = self.genetic_tracker.get_genealogy_tree(strategy_id)
        
        # Get evolution path
        evolution_path = self.genetic_tracker.get_evolution_path(strategy_id)
        
        # Calculate genetic similarity to other strategies
        similarity_scores = {}
        for other_id in self.listings.keys():
            if other_id != strategy_id:
                similarity = self.genetic_tracker.get_genetic_similarity(strategy_id, other_id)
                if similarity > 0.1:  # Only include if somewhat similar
                    similarity_scores[other_id] = similarity
        
        return {
            'strategy_id': strategy_id,
            'listing_info': {
                'name': listing.strategy_name,
                'material': listing.material,
                'operation_type': listing.operation_type,
                'author': listing.author,
                'timestamp': listing.timestamp.isoformat(),
                'tags': listing.tags
            },
            'survivor_badge': {
                'survivor_score': listing.survivor_badge.survivor_score,
                'badge_level': listing.survivor_badge.badge_level,
                'complexity_factor': listing.survivor_badge.complexity_factor,
                'environmental_resilience': listing.survivor_badge.environmental_resilience,
                'failure_history': listing.survivor_badge.failure_history
            },
            'genetic_lineage_tree': lineage,
            'evolution_path': [
                {
                    'generation': i+1,
                    'mutation_id': mutation.mutation_id,
                    'mutation_type': mutation.mutation_type.value,
                    'description': mutation.mutation_description,
                    'improvement': mutation.improvement_metric,
                    'timestamp': mutation.timestamp.isoformat()
                } for i, mutation in enumerate(evolution_path)
            ],
            'genetic_similarity_to_others': similarity_scores,
            'related_strategies': [sid for sid, sim in similarity_scores.items() if sim > 0.5],
            'genetic_diversity': listing.genetic_lineage.genetic_diversity,
            'branching_factor': listing.genetic_lineage.branching_factor,
            'query_timestamp': datetime.utcnow().isoformat()
        }
    
    def run_marketplace_simulation(self, duration_hours: float = 1.0) -> Dict[str, Any]:
        """
        Run a simulation of the marketplace to test resilience under stress conditions.
        
        Args:
            duration_hours: Duration of the simulation in hours
            
        Returns:
            Dictionary with simulation results
        """
        print(f"[MARKETPLACE] Running simulation for {duration_hours} hours...")
        
        # This would normally run the Nightmare Training protocol
        # For now, we'll simulate by evaluating all strategies under stress tests
        simulated_results = {
            'duration_hours': duration_hours,
            'strategies_evaluated': len(self.listings),
            'stress_tests_performed': 0,
            'survivor_badges_awarded': 0,
            'strategies_improved': 0,
            'economic_impact': 0.0,
            'simulation_start': datetime.utcnow().isoformat(),
            'simulation_end': None,
            'top_performers': []
        }
        
        # Simulate stress testing for each strategy
        stress_test_results = []
        for strategy_id, listing in self.listings.items():
            # Generate random stress tests for this strategy
            import random
            num_tests = random.randint(3, 10)
            
            for i in range(num_tests):
                # Simulate a stress test
                success = random.random() > (0.2 * (1 - listing.survivor_badge.survivor_score))
                
                stress_test_results.append({
                    'strategy_id': strategy_id,
                    'test_number': i+1,
                    'success': success,
                    'stress_level': random.uniform(0.5, 1.0),
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                simulated_results['stress_tests_performed'] += 1
                
                if not success:
                    # If the strategy failed the stress test, consider it for improvement
                    if random.random() > 0.7:  # 30% chance of improvement after failure
                        simulated_results['strategies_improved'] += 1
        
        # Calculate economic impact
        total_value = sum(listing.economic_value for listing in self.listings.values())
        simulated_results['economic_impact'] = total_value * duration_hours * 0.1  # Arbitrary multiplier
        
        # Identify top performers based on survivor score and economic value
        top_performers = sorted(
            self.listings.items(),
            key=lambda x: (x[1].survivor_badge.survivor_score + x[1].economic_value) / 2,
            reverse=True
        )[:10]
        
        simulated_results['top_performers'] = [
            {
                'strategy_id': sid,
                'name': listing.strategy_name,
                'survivor_score': listing.survivor_badge.survivor_score,
                'economic_value': listing.economic_value,
                'badge_level': listing.survivor_badge.badge_level
            } for sid, listing in top_performers
        ]
        
        simulated_results['simulation_end'] = datetime.utcnow().isoformat()
        
        print(f"[MARKETPLACE] Simulation completed. {simulated_results['stress_tests_performed']} tests performed.")
        
        return simulated_results
    
    def generate_marketplace_report(self) -> str:
        """
        Generate a comprehensive report about the marketplace status.
        
        Returns:
            JSON string with marketplace report
        """
        marketplace_stats = self.get_marketplace_statistics()
        
        report = {
            'marketplace_report': {
                'overview': {
                    'total_strategies': marketplace_stats['total_strategies'],
                    'average_survivor_score': marketplace_stats['average_survivor_score'],
                    'average_economic_value': marketplace_stats['average_economic_value'],
                    'top_materials': marketplace_stats['top_materials'],
                    'top_operations': marketplace_stats['top_operations'],
                    'growth_rate': marketplace_stats['strategy_growth_rate']
                },
                'rankings': {
                    'top_10_strategies': [
                        {
                            'strategy_id': listing.strategy_id,
                            'name': listing.strategy_name,
                            'material': listing.material,
                            'operation': listing.operation_type,
                            'survivor_score': listing.survivor_badge.survivor_score,
                            'economic_value': listing.economic_value,
                            'badge_level': listing.survivor_badge.badge_level,
                            'tags': listing.tags
                        } for listing in self.get_top_strategies(limit=10)
                    ]
                },
                'distribution': {
                    'badge_levels': marketplace_stats['badge_distribution'],
                    'score_ranges': marketplace_stats['score_distribution']
                },
                'economic_impact': {
                    'total_economic_value': sum(l.economic_value for l in self.listings.values()),
                    'savings_potential': 0.0,  # Placeholder - would be calculated from economic_auditor
                    'roi_metrics': self.economic_auditor.get_fleet_roi_metrics() if hasattr(self.economic_auditor, 'get_fleet_roi_metrics') else {}
                },
                'genetic_diversity': {
                    'average_diversity': sum(l.genetic_lineage.genetic_diversity for l in self.listings.values()) / len(self.listings) if self.listings else 0.0,
                    'most_diverse_lineages': self._get_most_diverse_lineages()
                },
                'report_generated': datetime.utcnow().isoformat()
            }
        }
        
        return json.dumps(report, indent=2)
    
    def _get_most_diverse_lineages(self) -> List[Dict[str, Any]]:
        """Get the most genetically diverse lineages in the marketplace."""
        sorted_lineages = sorted(
            self.listings.values(),
            key=lambda l: l.genetic_lineage.genetic_diversity,
            reverse=True
        )
        
        return [
            {
                'strategy_id': listing.strategy_id,
                'name': listing.strategy_name,
                'genetic_diversity': listing.genetic_lineage.genetic_diversity,
                'generation_count': listing.genetic_lineage.generation_count,
                'branching_factor': listing.genetic_lineage.branching_factor
            } for listing in sorted_lineages[:5]
        ]


# Example usage and testing
if __name__ == "__main__":
    print("Anti-Fragile Marketplace initialized successfully.")
    print("Ready to rank G-Code strategies by resilience rather than speed.")
    
    # Example usage would be:
    # marketplace = AntiFragileMarketplace()
    # 
    # # Submit a strategy to the marketplace
    # listing = marketplace.submit_strategy(
    #     strategy_id="STRAT_INCONEL_FACE_MILL_001",
    #     strategy_name="Inconel Face Mill - Conservative Approach",
    #     material="Inconel-718",
    #     operation_type="face_mill",
    #     parameters={"feed_rate": 1500, "rpm": 4000, "depth": 1.0},
    #     author="CNC_Operator_001",
    #     description="Conservative parameters for machining Inconel-718 to minimize tool wear"
    # )
    # 
    # # Get top strategies for Inconel milling
    # top_strategies = marketplace.get_top_strategies(
    #     material="Inconel-718", 
    #     operation_type="face_mill", 
    #     min_survivor_score=0.5,
    #     limit=5
    # )
    # 
    # print(f"Top Inconel face mill strategies:")
    # for i, strategy in enumerate(top_strategies, 1):
    #     print(f"  {i}. {strategy.strategy_name} - Score: {strategy.survivor_badge.survivor_score:.3f}")
    # 
    # # Generate marketplace report
    # report = marketplace.generate_marketplace_report()
    # print("\nMarketplace report generated successfully.")
    # 
    # # Run simulation
    # simulation_results = marketplace.run_marketplace_simulation(duration_hours=0.5)
    # print(f"Simulation completed with {simulation_results['strategies_evaluated']} strategies evaluated.")