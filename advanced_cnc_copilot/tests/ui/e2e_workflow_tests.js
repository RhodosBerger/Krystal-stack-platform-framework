/**
 * End-to-End Workflow Tests for FANUC RISE v2.1 Advanced CNC Copilot System
 * These tests validate complete user journeys and workflows across multiple components
 */

import { test, expect, describe, beforeAll, afterAll, beforeEach } from 'vitest';
import puppeteer from 'puppeteer';
import axios from 'axios';

// Mock API responses for testing
jest.mock('axios');

describe('FANUC RISE v2.1 E2E Workflows', () => {
  let browser;
  let page;

  beforeAll(async () => {
    // Launch browser for E2E tests
    browser = await puppeteer.launch({
      headless: true, // Set to false for debugging
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    page = await browser.newPage();
    
    // Set viewport for consistent testing
    await page.setViewport({ width: 1280, height: 720 });
  });

  afterAll(async () => {
    // Close browser after all tests
    if (browser) {
      await browser.close();
    }
  });

  beforeEach(async () => {
    // Reset any state before each test
    jest.clearAllMocks();
  });

  describe('Operator Workflow', () => {
    test('Complete operator workflow from dashboard to job completion', async () => {
      // Mock API responses for the operator workflow
      axios.get.mockResolvedValueOnce({
        data: {
          machines: [
            { id: 1, name: 'CNC-001', status: 'IDLE', spindleLoad: 0, temperature: 25, vibration: 0.1 },
            { id: 2, name: 'CNC-002', status: 'RUNNING', spindleLoad: 65, temperature: 38, vibration: 0.4 }
          ],
          overallMetrics: {
            uptime: 98.5,
            efficiency: 92.3,
            quality: 99.2,
            safety: 100
          }
        }
      });

      // Navigate to the operator dashboard
      await page.goto('http://localhost:3000/operator-dashboard', { waitUntil: 'networkidle2' });
      
      // Verify dashboard loaded
      await expect(page.title()).resolves.toMatch(/Operator Dashboard/);
      
      // Check that machine status is displayed
      await page.waitForSelector('[data-testid="machine-status"]');
      const machineStatus = await page.$eval('[data-testid="machine-status"]', el => el.textContent);
      expect(machineStatus).toContain('IDLE');
      
      // Select a machine to operate
      await page.click('[data-testid="select-machine-CNC-001"]');
      
      // Verify machine selection
      await page.waitForSelector('[data-testid="selected-machine"]');
      const selectedMachine = await page.$eval('[data-testid="selected-machine"]', el => el.textContent);
      expect(selectedMachine).toContain('CNC-001');
      
      // Start a new job
      await page.click('[data-testid="start-new-job"]');
      
      // Fill in job parameters
      await page.type('#job-name', 'Test Job 001');
      await page.type('#material-type', 'Aluminum 6061');
      await page.type('#feed-rate', '2500');
      await page.type('#rpm', '5000');
      await page.type('#depth-of-cut', '2.0');
      
      // Submit the job
      await page.click('[data-testid="submit-job"]');
      
      // Wait for job submission
      await page.waitForSelector('[data-testid="job-status-running"]');
      const jobStatus = await page.$eval('[data-testid="job-status-running"]', el => el.textContent);
      expect(jobStatus).toContain('RUNNING');
      
      // Monitor job progress
      await page.waitForSelector('[data-testid="progress-bar"]');
      const progressValue = await page.$eval('[data-testid="progress-bar"]', el => el.getAttribute('value'));
      expect(parseInt(progressValue)).toBeGreaterThanOrEqual(0);
      
      // Simulate job completion
      // In a real test, this would wait for the actual job to complete
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Verify job completion
      await page.waitForSelector('[data-testid="job-status-complete"]');
      const completionStatus = await page.$eval('[data-testid="job-status-complete"]', el => el.textContent);
      expect(completionStatus).toContain('COMPLETE');
      
      // Check that completion metrics are displayed
      await page.waitForSelector('[data-testid="completion-metrics"]');
      const metrics = await page.$eval('[data-testid="completion-metrics"]', el => el.textContent);
      expect(metrics).toContain('Quality');
      expect(metrics).toContain('Efficiency');
    });

    test('Emergency stop and recovery workflow', async () => {
      // Mock emergency scenario
      axios.get.mockResolvedValue({
        data: {
          machineId: 1,
          status: 'RUNNING',
          spindleLoad: 95, // Very high - approaching dangerous levels
          temperature: 68, // High temperature
          vibration: 1.8,  // High vibration
          emergencyStop: false
        }
      });

      await page.goto('http://localhost:3000/operator-dashboard', { waitUntil: 'networkidle2' });
      
      // Simulate high stress scenario
      await page.waitForSelector('[data-testid="neuro-safety-indicator"]');
      const stressIndicator = await page.$eval('[data-testid="neuro-safety-indicator"]', el => el.className);
      expect(stressIndicator).toContain('high-stress');
      
      // Check for automatic safety warnings
      await page.waitForSelector('[data-testid="safety-alert"]');
      const alertText = await page.$eval('[data-testid="safety-alert"]', el => el.textContent);
      expect(alertText).toContain('HIGH STRESS');
      
      // Verify emergency stop button is available
      await page.waitForSelector('[data-testid="emergency-stop"]');
      const emergencyStopBtn = await page.$('[data-testid="emergency-stop"]');
      expect(emergencyStopBtn).toBeTruthy();
      
      // Trigger emergency stop
      await page.click('[data-testid="emergency-stop"]');
      
      // Wait for emergency stop confirmation
      await page.waitForSelector('[data-testid="emergency-stop-confirmed"]');
      const emergencyStatus = await page.$eval('[data-testid="emergency-stop-confirmed"]', el => el.textContent);
      expect(emergencyStatus).toContain('EMERGENCY STOP');
      
      // Check that safety protocols activated
      await page.waitForSelector('[data-testid="safety-protocols-active"]');
      const safetyProtocols = await page.$eval('[data-testid="safety-protocols-active"]', el => el.textContent);
      expect(safetyProtocols).toContain('SAFETY PROTOCOLS');
      
      // Verify machine status changed to stopped
      await page.waitForSelector('[data-testid="machine-status"]');
      const machineStatus = await page.$eval('[data-testid="machine-status"]', el => el.textContent);
      expect(machineStatus).toContain('STOPPED');
    });
  });

  describe('Manager Workflow', () => {
    test('Fleet management and optimization workflow', async () => {
      // Mock fleet data
      axios.get.mockResolvedValue({
        data: {
          fleet: [
            {
              id: 'M001',
              name: 'CNC-001',
              status: 'RUNNING',
              utilization: 85,
              efficiency: 92,
              quality: 98,
              healthScore: 95
            },
            {
              id: 'M002',
              name: 'CNC-002',
              status: 'IDLE',
              utilization: 15,
              efficiency: 0,
              quality: 100,
              healthScore: 98
            },
            {
              id: 'M003',
              name: 'CNC-003',
              status: 'MAINTENANCE',
              utilization: 0,
              efficiency: 0,
              quality: 100,
              healthScore: 65
            }
          ],
          overallMetrics: {
            fleetUtilization: 45,
            fleetEfficiency: 78,
            fleetQuality: 97
          }
        }
      });

      await page.goto('http://localhost:3000/manager-dashboard', { waitUntil: 'networkidle2' });
      
      // Verify fleet overview loads
      await page.waitForSelector('[data-testid="fleet-overview"]');
      const fleetOverview = await page.$eval('[data-testid="fleet-overview"]', el => el.textContent);
      expect(fleetOverview).toContain('Fleet Status');
      
      // Check fleet metrics
      await page.waitForSelector('[data-testid="fleet-utilization"]');
      const utilization = await page.$eval('[data-testid="fleet-utilization"]', el => el.textContent);
      expect(utilization).toContain('45%');
      
      await page.waitForSelector('[data-testid="fleet-efficiency"]');
      const efficiency = await page.$eval('[data-testid="fleet-efficiency"]', el => el.textContent);
      expect(efficiency).toContain('78%');
      
      // Identify underutilized machines
      await page.waitForSelector('[data-testid="underutilized-machines"]');
      const underutilizedCount = await page.$$eval('[data-testid="underutilized-machines"] .machine-card', els => els.length);
      expect(underutilizedCount).toBeGreaterThan(0);
      
      // Check for maintenance needs
      await page.waitForSelector('[data-testid="maintenance-needed"]');
      const maintenanceNeeded = await page.$eval('[data-testid="maintenance-needed"]', el => el.textContent);
      expect(maintenanceNeeded).toContain('M003');
      
      // Initiate fleet optimization
      await page.click('[data-testid="optimize-fleet"]');
      
      // Wait for optimization results
      await page.waitForSelector('[data-testid="optimization-results"]');
      const optimizationResults = await page.$eval('[data-testid="optimization-results"]', el => el.textContent);
      expect(optimizationResults).toContain('Optimization Complete');
      
      // Verify suggested actions are displayed
      await page.waitForSelector('[data-testid="suggested-actions"]');
      const suggestedActions = await page.$eval('[data-testid="suggested-actions"]', el => el.textContent);
      expect(suggestedActions).toContain('Rebalance Load');
      expect(suggestedActions).toContain('Schedule Maintenance');
    });

    test('Cross-machine learning and pattern recognition', async () => {
      // Mock cross-machine data
      axios.get.mockResolvedValue({
        data: {
          patterns: [
            {
              id: 'P001',
              type: 'tool-wear',
              machines: ['M001', 'M002'],
              similarity: 0.89,
              recommendation: 'Reduce feed rate by 10%'
            },
            {
              id: 'P002',
              type: 'vibration-anomaly',
              machines: ['M003'],
              similarity: 0.95,
              recommendation: 'Check spindle alignment'
            }
          ],
          collectiveLearning: {
            totalPatterns: 12,
            appliedRecommendations: 8,
            preventedIssues: 3
          }
        }
      });

      await page.goto('http://localhost:3000/swarm-intelligence', { waitUntil: 'networkidle2' });
      
      // Verify swarm intelligence dashboard loads
      await page.waitForSelector('[data-testid="swarm-dashboard"]');
      const swarmTitle = await page.$eval('[data-testid="swarm-dashboard"]', el => el.textContent);
      expect(swarmTitle).toContain('Swarm Intelligence');
      
      // Check for identified patterns
      await page.waitForSelector('[data-testid="identified-patterns"]');
      const patternsCount = await page.$eval('[data-testid="identified-patterns"]', el => el.textContent);
      expect(patternsCount).toContain('12 Patterns');
      
      // Verify pattern details are displayed
      await page.waitForSelector('[data-testid="pattern-list"]');
      const patternRows = await page.$$('.pattern-row');
      expect(patternRows.length).toBeGreaterThan(0);
      
      // Check for collective learning metrics
      await page.waitForSelector('[data-testid="collective-learning"]');
      const collectiveLearning = await page.$eval('[data-testid="collective-learning"]', el => el.textContent);
      expect(collectiveLearning).toContain('Applied Recommendations: 8');
      expect(collectiveLearning).toContain('Prevented Issues: 3');
      
      // Apply a recommendation
      await page.click('[data-testid="apply-recommendation-P001"]');
      
      // Verify recommendation application
      await page.waitForSelector('[data-testid="recommendation-applied"]');
      const appliedRec = await page.$eval('[data-testid="recommendation-applied"]', el => el.textContent);
      expect(appliedRec).toContain('Recommendation Applied');
    });
  });

  describe('Creator Workflow', () => {
    test('Generative design and optimization workflow', async () => {
      // Mock creative studio data
      axios.get.mockResolvedValue({
        data: {
          designHistory: [
            { id: 'D001', name: 'Bracket Design v1', created: '2023-07-15', status: 'APPROVED' },
            { id: 'D002', name: 'Housing Design v2', created: '2023-07-16', status: 'IN_PROGRESS' }
          ],
          optimizationSuggestions: [
            { id: 'O001', description: 'Reduce wall thickness by 15%', potentialSavings: '$240' },
            { id: 'O002', description: 'Modify corner radii for better tool path', potentialSavings: '$180' }
          ]
        }
      });

      await page.goto('http://localhost:3000/creative-studio', { waitUntil: 'networkidle2' });
      
      // Verify creative studio loads
      await page.waitForSelector('[data-testid="creative-studio"]');
      const studioTitle = await page.$eval('[data-testid="creative-studio"]', el => el.textContent);
      expect(studioTitle).toContain('Creative Studio');
      
      // Start a new design
      await page.click('[data-testid="new-design"]');
      
      // Upload design file
      await page.waitForSelector('#design-upload');
      // Note: File upload in Puppeteer requires special handling
      // This is a simplified example
      
      // Enter design parameters
      await page.type('#design-name', 'New Bracket Design');
      await page.type('#material', 'Steel');
      await page.type('#dimensions', '100x50x25');
      
      // Submit design for processing
      await page.click('[data-testid="process-design"]');
      
      // Wait for processing
      await page.waitForSelector('[data-testid="processing-complete"]');
      const processingStatus = await page.$eval('[data-testid="processing-complete"]', el => el.textContent);
      expect(processingStatus).toContain('Processing Complete');
      
      // Check optimization suggestions
      await page.waitForSelector('[data-testid="optimization-suggestions"]');
      const suggestionsCount = await page.$$('.optimization-suggestion');
      expect(suggestionsCount.length).toBeGreaterThan(0);
      
      // Apply an optimization
      await page.click('[data-testid="apply-optimization-O001"]');
      
      // Verify optimization applied
      await page.waitForSelector('[data-testid="optimization-applied"]');
      const appliedOpt = await page.$eval('[data-testid="optimization-applied"]', el => el.textContent);
      expect(appliedOpt).toContain('Optimization Applied');
      
      // Generate G-code
      await page.click('[data-testid="generate-gcode"]');
      
      // Wait for G-code generation
      await page.waitForSelector('[data-testid="gcode-generated"]');
      const gcodeStatus = await page.$eval('[data-testid="gcode-generated"]', el => el.textContent);
      expect(gcodeStatus).toContain('G-Code Generated');
      
      // Verify G-code preview is available
      await page.waitForSelector('[data-testid="gcode-preview"]');
      const gcodePreview = await page.$eval('[data-testid="gcode-preview"]', el => el.textContent);
      expect(gcodePreview).toContain('G00');
      expect(gcodePreview).toContain('G01');
    });

    test('Shadow Council approval workflow', async () => {
      // Mock shadow council data
      axios.get.mockResolvedValue({
        data: {
          proposals: [
            {
              id: 'P001',
              title: 'Aggressive Machining Parameters',
              submitted: '2023-07-15T10:30:00Z',
              status: 'UNDER_REVIEW',
              proposer: 'Creator Agent',
              agents: {
                creator: { status: 'APPROVED', reasoning: 'Efficiency gains justified' },
                auditor: { status: 'PENDING', reasoning: 'Validating physics constraints' },
                accountant: { status: 'PENDING', reasoning: 'Calculating economic impact' }
              }
            }
          ]
        }
      });

      await page.goto('http://localhost:3000/shadow-council', { waitUntil: 'networkidle2' });
      
      // Verify shadow council dashboard loads
      await page.waitForSelector('[data-testid="shadow-council"]');
      const councilTitle = await page.$eval('[data-testid="shadow-council"]', el => el.textContent);
      expect(councilTitle).toContain('Shadow Council');
      
      // Check pending proposals
      await page.waitForSelector('[data-testid="pending-proposals"]');
      const pendingCount = await page.$eval('[data-testid="pending-proposals"]', el => el.textContent);
      expect(pendingCount).toContain('1 Pending');
      
      // View proposal details
      await page.click('[data-testid="view-proposal-P001"]');
      
      // Verify proposal details display
      await page.waitForSelector('[data-testid="proposal-details"]');
      const proposalTitle = await page.$eval('[data-testid="proposal-details"]', el => el.textContent);
      expect(proposalTitle).toContain('Aggressive Machining Parameters');
      
      // Check agent statuses
      await page.waitForSelector('[data-testid="creator-status"]');
      const creatorStatus = await page.$eval('[data-testid="creator-status"]', el => el.textContent);
      expect(creatorStatus).toContain('APPROVED');
      
      await page.waitForSelector('[data-testid="auditor-status"]');
      const auditorStatus = await page.$eval('[data-testid="auditor-status"]', el => el.textContent);
      expect(auditorStatus).toContain('PENDING');
      
      await page.waitForSelector('[data-testid="accountant-status"]');
      const accountantStatus = await page.$eval('[data-testid="accountant-status"]', el => el.textContent);
      expect(accountantStatus).toContain('PENDING');
      
      // Simulate auditor approval
      axios.post.mockResolvedValue({
        data: {
          success: true,
          updatedProposal: {
            id: 'P001',
            agents: {
              creator: { status: 'APPROVED', reasoning: 'Efficiency gains justified' },
              auditor: { status: 'APPROVED', reasoning: 'Physics constraints satisfied' },
              accountant: { status: 'PENDING', reasoning: 'Calculating economic impact' }
            },
            finalStatus: 'PENDING_FINAL_APPROVAL'
          }
        }
      });
      
      await page.click('[data-testid="approve-auditor-P001"]');
      
      // Wait for status update
      await page.waitForSelector('[data-testid="auditor-status-approved"]');
      const updatedAuditorStatus = await page.$eval('[data-testid="auditor-status-approved"]', el => el.textContent);
      expect(updatedAuditorStatus).toContain('APPROVED');
      
      // Simulate accountant approval
      axios.post.mockResolvedValue({
        data: {
          success: true,
          updatedProposal: {
            id: 'P001',
            agents: {
              creator: { status: 'APPROVED', reasoning: 'Efficiency gains justified' },
              auditor: { status: 'APPROVED', reasoning: 'Physics constraints satisfied' },
              accountant: { status: 'APPROVED', reasoning: 'Positive economic impact' }
            },
            finalStatus: 'APPROVED'
          }
        }
      });
      
      await page.click('[data-testid="approve-accountant-P001"]');
      
      // Verify final approval
      await page.waitForSelector('[data-testid="proposal-approved"]');
      const finalStatus = await page.$eval('[data-testid="proposal-approved"]', el => el.textContent);
      expect(finalStatus).toContain('APPROVED');
    });
  });

  describe('System Integration Workflow', () => {
    test('Complete manufacturing workflow from design to delivery', async () => {
      // This test simulates the complete workflow from design creation to job completion
      
      // Step 1: Create a design in the creative studio
      await page.goto('http://localhost:3000/creative-studio', { waitUntil: 'networkidle2' });
      
      // Mock the necessary API calls for the entire workflow
      axios.get.mockImplementation((url) => {
        if (url.includes('/designs')) {
          return Promise.resolve({
            data: { designs: [] }
          });
        } else if (url.includes('/machines')) {
          return Promise.resolve({
            data: { 
              machines: [{ id: 1, name: 'CNC-001', status: 'IDLE' }],
              overallMetrics: { uptime: 95, efficiency: 90 }
            }
          });
        } else if (url.includes('/jobs')) {
          return Promise.resolve({
            data: { jobs: [], queue: [] }
          });
        } else if (url.includes('/telemetry')) {
          return Promise.resolve({
            data: { 
              current: { spindleLoad: 0, temperature: 25, vibration: 0.1 },
              history: []
            }
          });
        } else {
          return Promise.resolve({ data: {} });
        }
      });
      
      // Create a new design
      await page.click('[data-testid="new-design"]');
      await page.type('#design-name', 'Complete Workflow Test Design');
      await page.type('#material', 'Aluminum');
      await page.click('[data-testid="process-design"]');
      
      // Wait for design processing
      await page.waitForTimeout(1000);
      
      // Step 2: Submit design for manufacturing
      await page.click('[data-testid="submit-for-manufacturing"]');
      
      // Step 3: Navigate to operator dashboard
      await page.goto('http://localhost:3000/operator-dashboard', { waitUntil: 'networkidle2' });
      
      // Step 4: Select machine and start job
      await page.waitForSelector('[data-testid="select-machine-CNC-001"]');
      await page.click('[data-testid="select-machine-CNC-001"]');
      
      // Step 5: Start the job from the queued design
      await page.click('[data-testid="start-queued-job"]');
      
      // Step 6: Monitor job progress
      await page.waitForSelector('[data-testid="job-progress"]');
      await page.waitForFunction(() => {
        const progressEl = document.querySelector('[data-testid="job-progress"]');
        return progressEl && parseInt(progressEl.textContent) > 50;
      }, { timeout: 10000 }); // Wait up to 10 seconds for progress to reach 50%
      
      // Step 7: Complete the job
      // In a real test, we'd wait for the actual completion
      await page.waitForTimeout(2000);
      
      // Step 8: Verify completion metrics
      await page.waitForSelector('[data-testid="completion-report"]');
      const completionReport = await page.$eval('[data-testid="completion-report"]', el => el.textContent);
      expect(completionReport).toContain('Job Complete');
      expect(completionReport).toContain('Quality');
      expect(completionReport).toContain('Efficiency');
      
      // Step 9: Verify economic metrics
      await page.waitForSelector('[data-testid="economic-metrics"]');
      const economicMetrics = await page.$eval('[data-testid="economic-metrics"]', el => el.textContent);
      expect(economicMetrics).toContain('Profit');
      expect(economicMetrics).toContain('Cost');
    });

    test('Nightmare training and learning integration', async () => {
      // Mock nightmare training data
      axios.get.mockResolvedValue({
        data: {
          nightmareSessions: [
            {
              id: 'NS001',
              startTime: '2023-07-15T02:00:00Z',
              endTime: '2023-07-15T03:00:00Z',
              scenariosRun: 45,
              anomaliesDetected: 3,
              lessonsLearned: 2
            }
          ],
          learningOutcomes: [
            {
              id: 'LO001',
              description: 'Improved vibration detection sensitivity',
              appliedTo: ['M001', 'M002'],
              effectiveness: 0.85
            }
          ]
        }
      });

      await page.goto('http://localhost:3000/nightmare-training', { waitUntil: 'networkidle2' });
      
      // Verify nightmare training dashboard loads
      await page.waitForSelector('[data-testid="nightmare-training"]');
      const trainingTitle = await page.$eval('[data-testid="nightmare-training"]', el => el.textContent);
      expect(trainingTitle).toContain('Nightmare Training');
      
      // Check for recent sessions
      await page.waitForSelector('[data-testid="recent-sessions"]');
      const recentSessions = await page.$eval('[data-testid="recent-sessions"]', el => el.textContent);
      expect(recentSessions).toContain('NS001');
      
      // View session details
      await page.click('[data-testid="view-session-NS001"]');
      
      // Verify session metrics
      await page.waitForSelector('[data-testid="session-metrics"]');
      const sessionMetrics = await page.$eval('[data-testid="session-metrics"]', el => el.textContent);
      expect(sessionMetrics).toContain('45 Scenarios');
      expect(sessionMetrics).toContain('3 Anomalies');
      expect(sessionMetrics).toContain('2 Lessons');
      
      // Check learning outcomes
      await page.waitForSelector('[data-testid="learning-outcomes"]');
      const outcomesCount = await page.$eval('[data-testid="learning-outcomes"]', el => el.textContent);
      expect(outcomesCount).toContain('1 Outcome');
      
      // Verify outcome details
      await page.waitForSelector('[data-testid="outcome-LO001"]');
      const outcomeDetails = await page.$eval('[data-testid="outcome-LO001"]', el => el.textContent);
      expect(outcomeDetails).toContain('Improved vibration detection');
      expect(outcomeDetails).toContain('Effectiveness: 85%');
    });
  });

  describe('Responsive Design and Cross-Device Workflow', () => {
    test('Workflow works on tablet-sized screen', async () => {
      // Set tablet viewport
      await page.setViewport({ width: 768, height: 1024 });
      
      // Navigate to operator dashboard
      await page.goto('http://localhost:3000/operator-dashboard', { waitUntil: 'networkidle2' });
      
      // Verify key elements are accessible on tablet
      await page.waitForSelector('[data-testid="hamburger-menu"]');
      const menuBtn = await page.$('[data-testid="hamburger-menu"]');
      expect(menuBtn).toBeTruthy();
      
      // Test collapsible elements
      await page.click('[data-testid="hamburger-menu"]');
      await page.waitForSelector('[data-testid="mobile-nav"]');
      
      // Navigate to machine selection
      await page.click('[data-testid="mobile-nav-machines"]');
      await page.waitForSelector('[data-testid="machine-list"]');
      
      // Verify touch targets are appropriately sized
      const touchTargets = await page.$$('.touch-target');
      for (const target of touchTargets) {
        const boundingBox = await target.boundingBox();
        expect(boundingBox.width).toBeGreaterThanOrEqual(44); // Minimum touch target size
        expect(boundingBox.height).toBeGreaterThanOrEqual(44);
      }
    });

    test('Workflow works on mobile-sized screen', async () => {
      // Set mobile viewport
      await page.setViewport({ width: 375, height: 667 });
      
      // Navigate to dashboard
      await page.goto('http://localhost:3000/', { waitUntil: 'networkidle2' });
      
      // Test mobile-specific workflow
      await page.waitForSelector('[data-testid="mobile-login"]');
      await page.type('#mobile-username', 'operator');
      await page.type('#mobile-password', 'password');
      await page.click('[data-testid="mobile-login-btn"]');
      
      // Verify mobile dashboard loads
      await page.waitForSelector('[data-testid="mobile-dashboard"]');
      const mobileDashboard = await page.$eval('[data-testid="mobile-dashboard"]', el => el.textContent);
      expect(mobileDashboard).toBeTruthy();
      
      // Test swipe gestures for navigation (simulated)
      await page.tap('[data-testid="swipe-area"]');
      await page.waitForSelector('[data-testid="sidebar-open"]');
      
      // Navigate to job monitoring
      await page.click('[data-testid="mobile-nav-jobs"]');
      await page.waitForSelector('[data-testid="mobile-job-list"]');
      
      // Verify job status is clearly visible on small screen
      const jobStatus = await page.$eval('[data-testid="mobile-job-status"]', el => el.textContent);
      expect(jobStatus.length).toBeGreaterThan(0);
    });
  });

  describe('Error Recovery and Edge Cases', () => {
    test('Handles network interruption during job monitoring', async () => {
      // Mock intermittent network failure
      let requestCount = 0;
      axios.get.mockImplementation((url) => {
        requestCount++;
        if (requestCount % 3 === 0) {
          // Every third request fails
          return Promise.reject(new Error('Network error'));
        }
        return Promise.resolve({
          data: {
            jobId: 'J001',
            status: 'RUNNING',
            progress: 65,
            currentOperation: 'Milling',
            partsCompleted: 42,
            totalParts: 50
          }
        });
      });

      await page.goto('http://localhost:3000/job-monitor', { waitUntil: 'networkidle2' });
      
      // Allow some time for multiple requests
      await page.waitForTimeout(3000);
      
      // Verify error handling doesn't break the UI
      await page.waitForSelector('[data-testid="job-monitor"]');
      const monitorExists = await page.$eval('[data-testid="job-monitor"]', el => !!el);
      expect(monitorExists).toBe(true);
      
      // Check for graceful error messages
      const errorMessages = await page.$$('.error-message');
      // Should handle errors gracefully without crashing
      expect(errorMessages.length).toBeLessThan(10); // Should not flood with errors
    });

    test('Recovers from invalid data input', async () => {
      await page.goto('http://localhost:3000/operator-dashboard', { waitUntil: 'networkidle2' });
      
      // Try to submit invalid job parameters
      await page.type('#feed-rate', '-100'); // Invalid negative value
      await page.type('#rpm', '999999'); // Excessively high value
      
      // Attempt to submit
      await page.click('[data-testid="submit-job"]');
      
      // Verify validation errors are displayed
      await page.waitForSelector('[data-testid="validation-error"]');
      const validationErrors = await page.$$('.validation-error');
      expect(validationErrors.length).toBeGreaterThan(0);
      
      // Try with valid parameters after invalid ones
      await page.type('#feed-rate', '2500'); // Valid value
      await page.type('#rpm', '5000'); // Valid value
      
      // Submit should now succeed
      await page.click('[data-testid="submit-job-valid"]');
      await page.waitForSelector('[data-testid="job-submitted"]');
      const jobSubmitted = await page.$eval('[data-testid="job-submitted"]', el => el.textContent);
      expect(jobSubmitted).toContain('Submitted');
    });
  });
});

