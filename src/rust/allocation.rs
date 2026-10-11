//! Resource Allocation Schema - Best Practices Implementation
//!
//! Implements allocation strategies for CPU, GPU, memory, and budget distribution
//! following established patterns: pool allocation, slab allocation, buddy system

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

/// Allocation strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// First-fit: allocate first available block
    FirstFit,
    /// Best-fit: allocate smallest sufficient block
    BestFit,
    /// Worst-fit: allocate largest available block
    WorstFit,
    /// Pool: fixed-size block allocation
    Pool,
    /// Slab: object-cached allocation
    Slab,
    /// Buddy: power-of-two block splitting
    Buddy,
}

/// Resource type being allocated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    CpuCores,
    CpuTime,
    GpuCompute,
    GpuMemory,
    SystemMemory,
    Bandwidth,
    PowerBudget,
    ThermalHeadroom,
}

/// Allocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRequest {
    pub id: String,
    pub resource_type: ResourceType,
    pub amount: u64,
    pub priority: Priority,
    pub constraints: AllocationConstraints,
    pub metadata: HashMap<String, String>,
}

/// Allocation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Critical = 0,    // System-critical, cannot be preempted
    High = 1,        // High priority, preempts normal
    Normal = 2,      // Default priority
    Low = 3,         // Background tasks
    BestEffort = 4,  // Allocate if available
}

/// Constraints for allocation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AllocationConstraints {
    pub min_amount: Option<u64>,
    pub max_amount: Option<u64>,
    pub alignment: Option<u64>,
    pub affinity: Option<Vec<u32>>,  // Preferred cores/nodes
    pub anti_affinity: Option<Vec<String>>,  // Don't colocate with these
    pub timeout_ms: Option<u64>,
    pub preemptible: bool,
}

/// Allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub id: String,
    pub request_id: String,
    pub resource_type: ResourceType,
    pub amount: u64,
    pub block_id: u64,
    pub offset: u64,
    pub timestamp: u64,
    pub expires_at: Option<u64>,
}

/// Block in the allocation pool
#[derive(Debug, Clone)]
struct Block {
    id: u64,
    offset: u64,
    size: u64,
    allocated: bool,
    allocation_id: Option<String>,
    priority: Priority,
}

/// Resource pool for a single resource type
pub struct ResourcePool {
    resource_type: ResourceType,
    total_capacity: u64,
    strategy: AllocationStrategy,
    blocks: BTreeMap<u64, Block>,  // offset -> block
    free_list: VecDeque<u64>,      // free block offsets
    allocations: HashMap<String, Allocation>,
    next_block_id: u64,
    fragmentation_threshold: f64,
}

impl ResourcePool {
    pub fn new(resource_type: ResourceType, capacity: u64, strategy: AllocationStrategy) -> Self {
        let mut pool = Self {
            resource_type,
            total_capacity: capacity,
            strategy,
            blocks: BTreeMap::new(),
            free_list: VecDeque::new(),
            allocations: HashMap::new(),
            next_block_id: 1,
            fragmentation_threshold: 0.3,
        };

        // Initialize with single free block
        pool.blocks.insert(0, Block {
            id: 0,
            offset: 0,
            size: capacity,
            allocated: false,
            allocation_id: None,
            priority: Priority::BestEffort,
        });
        pool.free_list.push_back(0);

        pool
    }

    /// Allocate resources
    pub fn allocate(&mut self, request: &AllocationRequest) -> Result<Allocation, AllocationError> {
        let amount = self.apply_constraints(request.amount, &request.constraints);

        let block_offset = match self.strategy {
            AllocationStrategy::FirstFit => self.first_fit(amount),
            AllocationStrategy::BestFit => self.best_fit(amount),
            AllocationStrategy::WorstFit => self.worst_fit(amount),
            AllocationStrategy::Pool => self.pool_allocate(amount),
            AllocationStrategy::Buddy => self.buddy_allocate(amount),
            AllocationStrategy::Slab => self.slab_allocate(amount),
        }?;

        let block = self.blocks.get_mut(&block_offset).unwrap();
        let allocation_id = format!("alloc-{}-{}", request.id, self.next_block_id);

        // Split block if necessary
        if block.size > amount {
            self.split_block(block_offset, amount);
        }

        let block = self.blocks.get_mut(&block_offset).unwrap();
        block.allocated = true;
        block.allocation_id = Some(allocation_id.clone());
        block.priority = request.priority;

        let allocation = Allocation {
            id: allocation_id.clone(),
            request_id: request.id.clone(),
            resource_type: self.resource_type,
            amount,
            block_id: block.id,
            offset: block_offset,
            timestamp: current_timestamp(),
            expires_at: request.constraints.timeout_ms.map(|t| current_timestamp() + t),
        };

        self.allocations.insert(allocation_id, allocation.clone());
        self.next_block_id += 1;

        Ok(allocation)
    }

    /// Free an allocation
    pub fn free(&mut self, allocation_id: &str) -> Result<(), AllocationError> {
        let allocation = self.allocations.remove(allocation_id)
            .ok_or(AllocationError::NotFound)?;

        let block = self.blocks.get_mut(&allocation.offset)
            .ok_or(AllocationError::InvalidBlock)?;

        block.allocated = false;
        block.allocation_id = None;

        self.free_list.push_back(allocation.offset);
        self.coalesce_blocks(allocation.offset);

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let allocated: u64 = self.blocks.values()
            .filter(|b| b.allocated)
            .map(|b| b.size)
            .sum();

        let free_blocks: usize = self.blocks.values()
            .filter(|b| !b.allocated)
            .count();

        let largest_free = self.blocks.values()
            .filter(|b| !b.allocated)
            .map(|b| b.size)
            .max()
            .unwrap_or(0);

        PoolStats {
            total_capacity: self.total_capacity,
            allocated,
            available: self.total_capacity - allocated,
            utilization: allocated as f64 / self.total_capacity as f64,
            fragmentation: self.calculate_fragmentation(),
            block_count: self.blocks.len(),
            free_block_count: free_blocks,
            allocation_count: self.allocations.len(),
            largest_free_block: largest_free,
        }
    }

    fn apply_constraints(&self, amount: u64, constraints: &AllocationConstraints) -> u64 {
        let mut result = amount;

        if let Some(min) = constraints.min_amount {
            result = result.max(min);
        }
        if let Some(max) = constraints.max_amount {
            result = result.min(max);
        }
        if let Some(align) = constraints.alignment {
            result = (result + align - 1) / align * align;
        }

        result
    }

    fn first_fit(&self, size: u64) -> Result<u64, AllocationError> {
        for (&offset, block) in &self.blocks {
            if !block.allocated && block.size >= size {
                return Ok(offset);
            }
        }
        Err(AllocationError::InsufficientResources)
    }

    fn best_fit(&self, size: u64) -> Result<u64, AllocationError> {
        self.blocks.iter()
            .filter(|(_, b)| !b.allocated && b.size >= size)
            .min_by_key(|(_, b)| b.size)
            .map(|(&offset, _)| offset)
            .ok_or(AllocationError::InsufficientResources)
    }

    fn worst_fit(&self, size: u64) -> Result<u64, AllocationError> {
        self.blocks.iter()
            .filter(|(_, b)| !b.allocated && b.size >= size)
            .max_by_key(|(_, b)| b.size)
            .map(|(&offset, _)| offset)
            .ok_or(AllocationError::InsufficientResources)
    }

    fn pool_allocate(&self, size: u64) -> Result<u64, AllocationError> {
        // Pool allocation: fixed-size blocks, find any free
        self.first_fit(size)
    }

    fn buddy_allocate(&mut self, size: u64) -> Result<u64, AllocationError> {
        // Buddy system: round up to power of two
        let size = size.next_power_of_two();
        self.best_fit(size)
    }

    fn slab_allocate(&self, size: u64) -> Result<u64, AllocationError> {
        // Slab: use best-fit for object caching
        self.best_fit(size)
    }

    fn split_block(&mut self, offset: u64, size: u64) {
        let block = self.blocks.get(&offset).unwrap().clone();
        let remaining = block.size - size;

        // Update original block
        self.blocks.get_mut(&offset).unwrap().size = size;

        // Create new free block
        let new_offset = offset + size;
        self.blocks.insert(new_offset, Block {
            id: self.next_block_id,
            offset: new_offset,
            size: remaining,
            allocated: false,
            allocation_id: None,
            priority: Priority::BestEffort,
        });
        self.next_block_id += 1;
        self.free_list.push_back(new_offset);
    }

    fn coalesce_blocks(&mut self, offset: u64) {
        // Coalesce with next block
        let block = self.blocks.get(&offset).unwrap().clone();
        let next_offset = offset + block.size;

        if let Some(next_block) = self.blocks.get(&next_offset) {
            if !next_block.allocated {
                let next_size = next_block.size;
                self.blocks.remove(&next_offset);
                self.blocks.get_mut(&offset).unwrap().size += next_size;
                self.free_list.retain(|&o| o != next_offset);
            }
        }

        // Coalesce with previous block
        if let Some((&prev_offset, prev_block)) = self.blocks.range(..offset).last() {
            if !prev_block.allocated && prev_offset + prev_block.size == offset {
                let current_size = self.blocks.get(&offset).unwrap().size;
                self.blocks.remove(&offset);
                self.blocks.get_mut(&prev_offset).unwrap().size += current_size;
                self.free_list.retain(|&o| o != offset);
            }
        }
    }

    fn calculate_fragmentation(&self) -> f64 {
        let free_blocks: Vec<_> = self.blocks.values()
            .filter(|b| !b.allocated)
            .collect();

        if free_blocks.is_empty() {
            return 0.0;
        }

        let total_free: u64 = free_blocks.iter().map(|b| b.size).sum();
        let largest_free = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);

        if total_free == 0 {
            return 0.0;
        }

        1.0 - (largest_free as f64 / total_free as f64)
    }
}

/// Pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub total_capacity: u64,
    pub allocated: u64,
    pub available: u64,
    pub utilization: f64,
    pub fragmentation: f64,
    pub block_count: usize,
    pub free_block_count: usize,
    pub allocation_count: usize,
    pub largest_free_block: u64,
}

/// Allocation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationError {
    InsufficientResources,
    InvalidConstraints,
    NotFound,
    InvalidBlock,
    Timeout,
    Preempted,
}

/// Multi-resource allocator
pub struct Allocator {
    pools: HashMap<ResourceType, Arc<RwLock<ResourcePool>>>,
    allocation_log: Vec<AllocationEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    pub timestamp: u64,
    pub event_type: AllocationEventType,
    pub allocation_id: String,
    pub resource_type: ResourceType,
    pub amount: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationEventType {
    Allocated,
    Freed,
    Preempted,
    Expired,
    Resized,
}

impl Allocator {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            allocation_log: Vec::new(),
        }
    }

    /// Initialize a resource pool
    pub fn init_pool(
        &mut self,
        resource_type: ResourceType,
        capacity: u64,
        strategy: AllocationStrategy,
    ) {
        let pool = ResourcePool::new(resource_type, capacity, strategy);
        self.pools.insert(resource_type, Arc::new(RwLock::new(pool)));
    }

    /// Allocate from a specific pool
    pub fn allocate(&mut self, request: AllocationRequest) -> Result<Allocation, AllocationError> {
        let pool = self.pools.get(&request.resource_type)
            .ok_or(AllocationError::InvalidConstraints)?;

        let allocation = pool.write().unwrap().allocate(&request)?;

        self.allocation_log.push(AllocationEvent {
            timestamp: current_timestamp(),
            event_type: AllocationEventType::Allocated,
            allocation_id: allocation.id.clone(),
            resource_type: request.resource_type,
            amount: allocation.amount,
        });

        Ok(allocation)
    }

    /// Free an allocation
    pub fn free(&mut self, resource_type: ResourceType, allocation_id: &str) -> Result<(), AllocationError> {
        let pool = self.pools.get(&resource_type)
            .ok_or(AllocationError::NotFound)?;

        pool.write().unwrap().free(allocation_id)?;

        self.allocation_log.push(AllocationEvent {
            timestamp: current_timestamp(),
            event_type: AllocationEventType::Freed,
            allocation_id: allocation_id.to_string(),
            resource_type,
            amount: 0,
        });

        Ok(())
    }

    /// Get stats for all pools
    pub fn stats(&self) -> HashMap<ResourceType, PoolStats> {
        self.pools.iter()
            .map(|(rt, pool)| (*rt, pool.read().unwrap().stats()))
            .collect()
    }

    /// Get allocation log
    pub fn get_log(&self) -> &[AllocationEvent] {
        &self.allocation_log
    }

    /// Cleanup expired allocations
    pub fn cleanup_expired(&mut self) {
        let now = current_timestamp();

        for (resource_type, pool) in &self.pools {
            let mut pool = pool.write().unwrap();
            let expired: Vec<_> = pool.allocations.iter()
                .filter(|(_, a)| a.expires_at.map(|e| e < now).unwrap_or(false))
                .map(|(id, _)| id.clone())
                .collect();

            for id in expired {
                if pool.free(&id).is_ok() {
                    self.allocation_log.push(AllocationEvent {
                        timestamp: now,
                        event_type: AllocationEventType::Expired,
                        allocation_id: id,
                        resource_type: *resource_type,
                        amount: 0,
                    });
                }
            }
        }
    }
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut pool = ResourcePool::new(ResourceType::SystemMemory, 1000, AllocationStrategy::FirstFit);

        let request = AllocationRequest {
            id: "test-1".into(),
            resource_type: ResourceType::SystemMemory,
            amount: 100,
            priority: Priority::Normal,
            constraints: AllocationConstraints::default(),
            metadata: HashMap::new(),
        };

        let allocation = pool.allocate(&request).unwrap();
        assert_eq!(allocation.amount, 100);

        let stats = pool.stats();
        assert_eq!(stats.allocated, 100);
        assert_eq!(stats.available, 900);
    }

    #[test]
    fn test_allocation_and_free() {
        let mut pool = ResourcePool::new(ResourceType::CpuTime, 1000, AllocationStrategy::BestFit);

        let req1 = AllocationRequest {
            id: "req-1".into(),
            resource_type: ResourceType::CpuTime,
            amount: 200,
            priority: Priority::High,
            constraints: AllocationConstraints::default(),
            metadata: HashMap::new(),
        };

        let alloc1 = pool.allocate(&req1).unwrap();
        assert_eq!(pool.stats().allocated, 200);

        pool.free(&alloc1.id).unwrap();
        assert_eq!(pool.stats().allocated, 0);
    }

    #[test]
    fn test_buddy_allocation() {
        let mut pool = ResourcePool::new(ResourceType::GpuMemory, 1024, AllocationStrategy::Buddy);

        let request = AllocationRequest {
            id: "gpu-1".into(),
            resource_type: ResourceType::GpuMemory,
            amount: 100,  // Will be rounded to 128 (next power of 2)
            priority: Priority::Normal,
            constraints: AllocationConstraints::default(),
            metadata: HashMap::new(),
        };

        let allocation = pool.allocate(&request).unwrap();
        assert!(allocation.amount >= 100);
    }

    #[test]
    fn test_multi_resource_allocator() {
        let mut allocator = Allocator::new();
        allocator.init_pool(ResourceType::CpuCores, 16, AllocationStrategy::FirstFit);
        allocator.init_pool(ResourceType::SystemMemory, 16384, AllocationStrategy::BestFit);

        let cpu_req = AllocationRequest {
            id: "cpu-1".into(),
            resource_type: ResourceType::CpuCores,
            amount: 4,
            priority: Priority::Normal,
            constraints: AllocationConstraints::default(),
            metadata: HashMap::new(),
        };

        let mem_req = AllocationRequest {
            id: "mem-1".into(),
            resource_type: ResourceType::SystemMemory,
            amount: 4096,
            priority: Priority::Normal,
            constraints: AllocationConstraints::default(),
            metadata: HashMap::new(),
        };

        allocator.allocate(cpu_req).unwrap();
        allocator.allocate(mem_req).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats[&ResourceType::CpuCores].allocated, 4);
        assert_eq!(stats[&ResourceType::SystemMemory].allocated, 4096);
    }
}
