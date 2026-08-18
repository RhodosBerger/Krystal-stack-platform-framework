use nexusgrid_system::{NexusGrid, CpuGovernor, ThreadSafeNexusGrid, GridUpdate, UpdateType};

#[tokio::main]
async fn main() {
    println!("NexusGrid System - Multi-Dimensional Resource Management");
    println!("=====================================================");
    
    // Create the NexusGrid system
    let mut nexus_grid = NexusGrid::new(
        (8, 8, 8),  // CPU grid dimensions: Time × Priority × Cores
        (4, 4, 2),  // GPU grid dimensions: Time × Priority × GPUs
        (4, 4, 2)   // Memory grid dimensions: Time × Priority × Memory Types
    );
    
    println!("✓ NexusGrid system initialized");
    println!("  - CPU Grid: {}×{}×{}", 
             nexus_grid.cpu_grid.dimensions.0, 
             nexus_grid.cpu_grid.dimensions.1, 
             nexus_grid.cpu_grid.dimensions.2);
    println!("  - GPU Grid: {}×{}×{}", 
             nexus_grid.gpu_grid.dimensions.0, 
             nexus_grid.gpu_grid.dimensions.1, 
             nexus_grid.gpu_grid.dimensions.2);
    println!("  - Memory Grid: {}×{}×{}", 
             nexus_grid.memory_grid.dimensions.0, 
             nexus_grid.memory_grid.dimensions.1, 
             nexus_grid.memory_grid.dimensions.2);
    
    // Display available GPUs
    println!("\nAvailable GPUs:");
    for gpu in &nexus_grid.gpu_grid.available_gpus {
        println!("  - GPU {}: {} ({}MB memory)", gpu.id, gpu.name, gpu.memory_total_mb);
    }
    
    // Coordinate resource allocation for a test position
    let test_position = (2, 3, 1); // Time=2, Priority=3, Resource=1
    println!("\nCoordinating resource allocation for position {:?}", test_position);
    
    nexus_grid.coordinate_resource_allocation(test_position);
    
    // Display the results
    let cpu_cell = &nexus_grid.cpu_grid.cells[test_position.0][test_position.1][test_position.2];
    let gpu_cell = &nexus_grid.gpu_grid.cells[test_position.0][test_position.1][test_position.2];
    let memory_cell = &nexus_grid.memory_grid.cells[test_position.0][test_position.1][test_position.2];
    
    println!("  CPU Grid Cell Results:");
    println!("    - Frequency: {} MHz", cpu_cell.frequency);
    println!("    - Governor: {:?}", cpu_cell.governor);
    println!("    - Task Priority: {}", cpu_cell.task_priority);
    
    println!("  GPU Grid Cell Results:");
    println!("    - Assigned GPU: {:?}", gpu_cell.assigned_gpu);
    println!("    - Priority: {}", gpu_cell.priority);
    println!("    - Memory Requirement: {} MB", gpu_cell.memory_requirement_mb);
    
    println!("  Memory Grid Cell Results:");
    println!("    - Bandwidth Requirement: {:.2} GB/s", memory_cell.bandwidth_requirement);
    println!("    - Cache Locality: {:.2}", memory_cell.cache_locality);
    
    // Demonstrate performance target optimization
    println!("\nOptimizing for 2.0x performance target:");
    nexus_grid.optimize_for_performance_target(2.0);
    
    // Test thread-safe version
    println!("\nTesting Thread-Safe NexusGrid:");
    let thread_safe_grid = ThreadSafeNexusGrid::new(nexus_grid);
    
    // Send some updates
    let update1 = GridUpdate {
        position: (1, 1, 1),
        update_type: UpdateType::CpuFrequency(3000), // 3 GHz
        value: 0.0,
    };
    
    let update2 = GridUpdate {
        position: (1, 1, 1),
        update_type: UpdateType::GovernorChange(CpuGovernor::Performance),
        value: 0.0,
    };
    
    let update3 = GridUpdate {
        position: (2, 2, 0),
        update_type: UpdateType::MemoryBandwidth(32.0),
        value: 0.0,
    };
    
    thread_safe_grid.send_update(update1).expect("Failed to send update 1");
    thread_safe_grid.send_update(update2).expect("Failed to send update 2");
    thread_safe_grid.send_update(update3).expect("Failed to send update 3");
    
    // Process the updates
    thread_safe_grid.process_updates().expect("Failed to process updates");
    
    println!("✓ All updates processed successfully");
    
    // Get current grid state
    let current_state = thread_safe_grid.get_grid_state().expect("Failed to get grid state");
    println!("✓ Retrieved current grid state with performance target: {:.1}x", current_state.performance_target);
    
    // Performance scaling demonstration
    println!("\nPerformance Scaling Demonstration:");
    let scaling_targets = vec![1.0, 1.5, 2.0, 3.0];
    
    for target in scaling_targets {
        println!("  Setting performance target to {:.1}x...", target);
        // For this demo, we'll just print what would happen
        match target {
            1.0 => println!("    Base configuration maintained"),
            1.5 => println!("    Moderate parameter enhancements enabled"),
            2.0 => println!("    Advanced CPU optimization parameters enabled"),
            3.0 => println!("    Advanced GPU switching parameters enabled"),
            _ => println!("    High-performance mode activated"),
        }
    }
    
    println!("\nNexusGrid System demonstration completed!");
    println!("The system successfully demonstrates:");
    println!("- CPU governor management with real-time optimization");
    println!("- GPU switching with intelligent resource allocation");
    println!("- Cross-grid coordination for optimal performance");
    println!("- Thread-safe multi-grid system with async support");
    println!("- Performance scaling with adaptive parameters");
}

// Test functions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nexusgrid_initialization() {
        let grid = NexusGrid::new((4, 4, 4), (2, 2, 2), (2, 2, 2));
        
        assert_eq!(grid.cpu_grid.dimensions, (4, 4, 4));
        assert_eq!(grid.gpu_grid.dimensions, (2, 2, 2));
        assert_eq!(grid.memory_grid.dimensions, (2, 2, 2));
        assert_eq!(grid.performance_target, 1.0);
    }
    
    #[test]
    fn test_cpu_governor_determination() {
        let mut cpu_grid = nexusgrid_system::CpuGrid::new((2, 2, 2));
        
        // Set high priority and load
        cpu_grid.cells[0][0][0].task_priority = 220;
        cpu_grid.cells[0][0][0].predicted_load = 0.9;
        cpu_grid.cells[0][0][0].thermal_headroom = 25.0;
        
        let governor = cpu_grid.determine_optimal_governor((0, 0, 0));
        assert_eq!(governor, CpuGovernor::Performance);
    }
    
    #[test]
    fn test_gpu_selection() {
        let gpus = vec![
            nexusgrid_system::GpuDevice {
                id: 0,
                name: "GPU0".to_string(),
                memory_total_mb: 2048,
                memory_used_mb: 100,
                compute_units: 1024,
                clock_mhz: 1500,
                power_limit_w: 150.0,
                temperature_c: 60.0,
                utilization_percent: 10.0,
                vram_bandwidth_gbps: 320.0,
            }
        ];
        
        let gpu_grid = nexusgrid_system::GpuGrid::new((2, 2, 1), gpus);
        let cpu_grid = nexusgrid_system::CpuGrid::new((2, 2, 1));
        
        let result = gpu_grid.select_optimal_gpu((0, 0, 0), &cpu_grid);
        assert_eq!(result, Some(0));
    }
}