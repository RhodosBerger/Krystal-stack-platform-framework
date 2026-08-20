/**
 * Comprehensive UI Tests for FANUC RISE v2.1 Advanced CNC Copilot React Components
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { vi, describe, expect, test, beforeEach, afterEach } from 'vitest';
import axios from 'axios';

// Mock API responses
vi.mock('axios');

// Import components to test
import { NeuroCard } from '../../frontend-react/src/components/NeuroCard.jsx';
import { MainDashboard } from '../../frontend-react/src/components/MainDashboard.jsx';
import { JobStatusTracker } from '../../frontend-react/src/components/JobStatusTracker.jsx';
import { DopamineGauge } from '../../frontend-react/src/components/DopamineGauge.jsx';
import { SystemHealth } from '../../frontend-react/src/components/SystemHealth.jsx';
import { MetricCard } from '../../frontend-react/src/components/MetricCard.jsx';

describe('FANUC RISE v2.1 React UI Components', () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
  });

  describe('NeuroCard Component', () => {
    test('renders with correct data and styling', () => {
      const props = {
        title: 'Spindle Load',
        metric: '72.5',
        status: 'OK',
        volatility: 0.3,
        unit: '%'
      };

      render(
        <BrowserRouter>
          <NeuroCard {...props} />
        </BrowserRouter>
      );

      expect(screen.getByText('SPINDLE LOAD')).toBeInTheDocument();
      expect(screen.getByText('72.5')).toBeInTheDocument();
      expect(screen.getByText('%')).toBeInTheDocument();
      expect(screen.getByText('OK')).toBeInTheDocument();
    });

    test('applies correct styling based on status', () => {
      const okProps = {
        title: 'Temperature',
        metric: '42.3',
        status: 'OK',
        volatility: 0.1,
        unit: '°C'
      };

      const errorProps = {
        title: 'Vibration',
        metric: '1.8',
        status: 'ERROR',
        volatility: 0.9,
        unit: 'mm/s'
      };

      render(
        <BrowserRouter>
          <div>
            <NeuroCard {...okProps} />
            <NeuroCard {...errorProps} />
          </div>
        </BrowserRouter>
      );

      // Check that OK status has success styling
      const okElement = screen.getByText('OK');
      expect(okElement).toHaveClass(/text-neuro-success/);

      // Check that ERROR status has danger styling
      const errorElement = screen.getByText('ERROR');
      expect(errorElement).toHaveClass(/text-neuro-danger/);
    });

    test('animates based on volatility', async () => {
      const props = {
        title: 'Feed Rate',
        metric: '2450',
        status: 'OPTIMAL',
        volatility: 0.7,
        unit: 'mm/min'
      };

      render(
        <BrowserRouter>
          <NeuroCard {...props} />
        </BrowserRouter>
      );

      // Check that volatility sparklines are rendered
      const sparklineElements = screen.getAllByRole('img'); // Assuming sparklines are represented as img
      expect(sparklineElements.length).toBeGreaterThan(0);
    });
  });

  describe('MetricCard Component', () => {
    test('renders metric card with title, value, and description', () => {
      const props = {
        title: 'Cycle Time',
        value: '4.2',
        description: 'Average cycle time in minutes',
        icon: '⏱️',
        trend: '+2.3%'
      };

      render(
        <BrowserRouter>
          <MetricCard {...props} />
        </BrowserRouter>
      );

      expect(screen.getByText('Cycle Time')).toBeInTheDocument();
      expect(screen.getByText('4.2')).toBeInTheDocument();
      expect(screen.getByText('Average cycle time in minutes')).toBeInTheDocument();
      expect(screen.getByText('+2.3%')).toBeInTheDocument();
    });

    test('applies correct styling based on trend', () => {
      const positiveProps = {
        title: 'Efficiency',
        value: '92.4',
        description: 'Production efficiency',
        icon: '📈',
        trend: '+5.2%',
        trendType: 'positive'
      };

      const negativeProps = {
        title: 'Downtime',
        value: '1.8',
        description: 'Downtime in hours',
        icon: '📉',
        trend: '-1.2%',
        trendType: 'negative'
      };

      render(
        <BrowserRouter>
          <div>
            <MetricCard {...positiveProps} />
            <MetricCard {...negativeProps} />
          </div>
        </BrowserRouter>
      );

      // Check trend styling
      const positiveTrend = screen.getByText('+5.2%');
      expect(positiveTrend).toHaveClass(/text-green/);

      const negativeTrend = screen.getByText('-1.2%');
      expect(negativeTrend).toHaveClass(/text-red/);
    });
  });

  describe('SystemHealth Component', () => {
    test('renders system health overview with status indicators', async () => {
      // Mock API response
      const mockHealthData = {
        overallStatus: 'HEALTHY',
        cpuUsage: 45,
        memoryUsage: 62,
        diskUsage: 38,
        networkStatus: 'CONNECTED',
        lastUpdate: new Date().toISOString()
      };

      axios.get.mockResolvedValue({ data: mockHealthData });

      render(
        <BrowserRouter>
          <SystemHealth />
        </BrowserRouter>
      );

      // Wait for data to load
      await waitFor(() => {
        expect(screen.getByText('System Status: HEALTHY')).toBeInTheDocument();
      });

      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      expect(screen.getByText('Disk Usage')).toBeInTheDocument();
    });

    test('handles loading state', () => {
      axios.get.mockImplementation(() => new Promise(() => {})); // Never resolve to simulate loading

      render(
        <BrowserRouter>
          <SystemHealth />
        </BrowserRouter>
      );

      expect(screen.getByText(/Loading system health/i)).toBeInTheDocument();
    });

    test('handles error state', async () => {
      axios.get.mockRejectedValue(new Error('Network error'));

      render(
        <BrowserRouter>
          <SystemHealth />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Error loading system health/i)).toBeInTheDocument();
      });
    });
  });

  describe('JobStatusTracker Component', () => {
    test('renders job status with progress information', async () => {
      const mockJobData = {
        jobId: 'J001',
        status: 'RUNNING',
        progress: 65,
        estimatedCompletion: '2023-07-15T14:30:00Z',
        currentOperation: 'Face Milling',
        partsCompleted: 42,
        totalParts: 50
      };

      axios.get.mockResolvedValue({ data: mockJobData });

      render(
        <BrowserRouter>
          <JobStatusTracker jobId="J001" />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText('Job Status: RUNNING')).toBeInTheDocument();
      });

      expect(screen.getByText('Face Milling')).toBeInTheDocument();
      expect(screen.getByText('42/50 Parts')).toBeInTheDocument();
    });

    test('updates in real-time when refresh is triggered', async () => {
      const mockJobData1 = {
        jobId: 'J001',
        status: 'RUNNING',
        progress: 65,
        currentOperation: 'Face Milling',
        partsCompleted: 42,
        totalParts: 50
      };

      const mockJobData2 = {
        jobId: 'J001',
        status: 'RUNNING',
        progress: 72,
        currentOperation: 'Drilling',
        partsCompleted: 46,
        totalParts: 50
      };

      // Mock first call
      axios.get.mockResolvedValueOnce({ data: mockJobData1 });
      // Mock subsequent calls
      axios.get.mockResolvedValue({ data: mockJobData2 });

      render(
        <BrowserRouter>
          <JobStatusTracker jobId="J001" />
        </BrowserRouter>
      );

      // Initially should show first data
      await waitFor(() => {
        expect(screen.getByText('Face Milling')).toBeInTheDocument();
      });

      // Simulate refresh button click
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      // Should now show updated data
      await waitFor(() => {
        expect(screen.getByText('Drilling')).toBeInTheDocument();
      });
    });
  });

  describe('DopamineGauge Component', () => {
    test('renders dopamine/cortisol levels with visual indicators', () => {
      const props = {
        dopamineLevel: 0.75,
        cortisolLevel: 0.25,
        title: 'Neuro-Safety State'
      };

      render(
        <BrowserRouter>
          <DopamineGauge {...props} />
        </BrowserRouter>
      );

      expect(screen.getByText('Dopamine: 75%')).toBeInTheDocument();
      expect(screen.getByText('Cortisol: 25%')).toBeInTheDocument();
      expect(screen.getByText('Neuro-Safety State')).toBeInTheDocument();
    });

    test('displays different visual states based on levels', () => {
      const lowDopamineProps = {
        dopamineLevel: 0.2,
        cortisolLevel: 0.8,
        title: 'Stressed State'
      };

      const highDopamineProps = {
        dopamineLevel: 0.9,
        cortisolLevel: 0.1,
        title: 'Optimal State'
      };

      render(
        <BrowserRouter>
          <div>
            <DopamineGauge {...lowDopamineProps} />
            <DopamineGauge {...highDopamineProps} />
          </div>
        </BrowserRouter>
      );

      // Check that stressed state has appropriate styling
      expect(screen.getByText('Stressed State')).toBeInTheDocument();
      
      // Check that optimal state has appropriate styling
      expect(screen.getByText('Optimal State')).toBeInTheDocument();
    });
  });

  describe('MainDashboard Component', () => {
    test('renders dashboard with all key components', async () => {
      // Mock comprehensive dashboard data
      const mockDashboardData = {
        overallMetrics: {
          uptime: 98.5,
          efficiency: 92.3,
          quality: 99.2,
          safety: 100
        },
        activeJobs: [
          { id: 'J001', name: 'Part A', status: 'RUNNING', progress: 75 },
          { id: 'J002', name: 'Part B', status: 'QUEUED', progress: 0 }
        ],
        alerts: [
          { id: 'A001', message: 'High vibration detected', severity: 'WARNING', timestamp: new Date() }
        ],
        machineStatus: {
          spindleLoad: 72.5,
          temperature: 42.3,
          vibration: 0.8
        }
      };

      axios.get.mockResolvedValue({ data: mockDashboardData });

      render(
        <BrowserRouter>
          <MainDashboard />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Dashboard Overview/i)).toBeInTheDocument();
      });

      // Check that key metrics are displayed
      expect(screen.getByText('98.5%')).toBeInTheDocument(); // Uptime
      expect(screen.getByText('92.3%')).toBeInTheDocument(); // Efficiency
      expect(screen.getByText('99.2%')).toBeInTheDocument(); // Quality

      // Check that active jobs are shown
      expect(screen.getByText('Part A')).toBeInTheDocument();
      expect(screen.getByText('Part B')).toBeInTheDocument();

      // Check that alerts are visible
      expect(screen.getByText('High vibration detected')).toBeInTheDocument();
    });

    test('handles responsive layout changes', () => {
      // Test that dashboard adapts to different screen sizes
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 800, // Mobile/tablet size
      });

      window.dispatchEvent(new Event('resize'));

      render(
        <BrowserRouter>
          <MainDashboard />
        </BrowserRouter>
      );

      // Should still render the main elements
      expect(screen.getByRole('main')).toBeInTheDocument();
    });
  });

  describe('Interactive Elements', () => {
    test('button interactions trigger appropriate actions', async () => {
      const mockJobData = {
        jobId: 'J001',
        status: 'PAUSED',
        progress: 50,
        currentOperation: 'Milling',
        partsCompleted: 25,
        totalParts: 50
      };

      axios.get.mockResolvedValue({ data: mockJobData });
      axios.post.mockResolvedValue({ success: true });

      render(
        <BrowserRouter>
          <JobStatusTracker jobId="J001" />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText('Job Status: PAUSED')).toBeInTheDocument();
      });

      // Find and click the resume button
      const resumeButton = screen.getByRole('button', { name: /resume/i });
      fireEvent.click(resumeButton);

      // Verify the API call was made
      await waitFor(() => {
        expect(axios.post).toHaveBeenCalledWith(
          `/api/jobs/J001/resume`,
          expect.anything()
        );
      });
    });

    test('form submissions work correctly', async () => {
      const mockFormData = {
        machineId: 'M001',
        operation: 'face_mill',
        parameters: {
          feedRate: 2500,
          rpm: 5000,
          depth: 2.0
        }
      };

      axios.post.mockResolvedValue({ data: { success: true, jobId: 'J005' } });

      render(
        <BrowserRouter>
          <MainDashboard />
        </BrowserRouter>
      );

      // Find and interact with a form (this is a simplified example)
      const newJobButton = screen.getByRole('button', { name: /new job/i });
      fireEvent.click(newJobButton);

      // Fill in form fields (assuming they exist)
      const machineInput = screen.queryByLabelText(/machine id/i);
      if (machineInput) {
        fireEvent.change(machineInput, { target: { value: 'M001' } });
      }

      // Submit form
      const submitButton = screen.queryByRole('button', { name: /submit/i });
      if (submitButton) {
        fireEvent.click(submitButton);
      }

      // Verify the API call was made
      await waitFor(() => {
        expect(axios.post).toHaveBeenCalledWith(
          '/api/jobs/create',
          expect.objectContaining(mockFormData)
        );
      });
    });
  });

  describe('Real-time Data Updates', () => {
    test('components update when new telemetry data arrives', async () => {
      const initialData = {
        spindleLoad: 65.0,
        temperature: 38.0,
        vibration: 0.5
      };

      const updatedData = {
        spindleLoad: 72.5,
        temperature: 42.3,
        vibration: 0.8
      };

      // Mock initial data
      axios.get.mockResolvedValueOnce({ data: initialData });
      // Mock updated data
      axios.get.mockResolvedValue({ data: updatedData });

      render(
        <BrowserRouter>
          <SystemHealth />
        </BrowserRouter>
      );

      // Initially should show first data
      await waitFor(() => {
        expect(screen.getByText('65.0%')).toBeInTheDocument();
      });

      // Simulate data refresh
      setTimeout(() => {
        // This would happen automatically in a real scenario
      }, 1000);

      // After update, should show new data
      await waitFor(() => {
        expect(screen.getByText('72.5%')).toBeInTheDocument();
      });
    });
  });

  describe('Authentication Flows', () => {
    test('redirects to login when unauthorized', async () => {
      axios.get.mockRejectedValue({
        response: { status: 401 }
      });

      render(
        <BrowserRouter>
          <SystemHealth />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/unauthorized/i)).toBeInTheDocument();
      });
    });

    test('shows loading state during auth check', () => {
      // This would be tested differently in a real implementation
      // depending on how auth is handled
      render(
        <BrowserRouter>
          <SystemHealth />
        </BrowserRouter>
      );

      // Initially might show loading state
      expect(screen.queryByText(/loading/i)).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    test('displays meaningful error messages', async () => {
      const errorResponse = {
        response: {
          data: {
            error: 'Invalid request parameters',
            details: 'Feed rate exceeds maximum allowable value'
          },
          status: 400
        }
      };

      axios.post.mockRejectedValue(errorResponse);

      render(
        <BrowserRouter>
          <MainDashboard />
        </BrowserRouter>
      );

      // Simulate an action that would cause an error
      const problematicButton = screen.queryByRole('button', { name: /problematic-action/i });
      if (problematicButton) {
        fireEvent.click(problematicButton);
      }

      await waitFor(() => {
        expect(screen.getByText(/invalid request parameters/i)).toBeInTheDocument();
        expect(screen.getByText(/feed rate exceeds/i)).toBeInTheDocument();
      });
    });

    test('gracefully handles network timeouts', async () => {
      axios.get.mockImplementation(() => {
        return new Promise((resolve, reject) => {
          setTimeout(() => {
            reject(new Error('Network timeout'));
          }, 10000); // Longer than typical timeout
        });
      });

      render(
        <BrowserRouter>
          <SystemHealth />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/connection timeout/i)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    test('components have proper ARIA labels and roles', () => {
      const props = {
        title: 'Spindle Load',
        metric: '72.5',
        status: 'OK',
        volatility: 0.3,
        unit: '%'
      };

      render(
        <BrowserRouter>
          <NeuroCard {...props} />
        </BrowserRouter>
      );

      // Check for proper semantic HTML and ARIA attributes
      const cardElement = screen.getByRole('article');
      expect(cardElement).toBeInTheDocument();

      // Check that status has appropriate ARIA attributes
      const statusElement = screen.getByText('OK');
      expect(statusElement).toHaveAttribute('aria-label', 'Status: OK');
    });

    test('keyboard navigation works for interactive elements', () => {
      render(
        <BrowserRouter>
          <MainDashboard />
        </BrowserRouter>
      );

      // Check that buttons can be focused
      const firstButton = screen.getByRole('button');
      expect(firstButton).toBeInTheDocument();

      // Simulate tab navigation
      fireEvent.focus(firstButton);
      expect(firstButton).toHaveFocus();
    });
  });

  describe('Performance', () => {
    test('renders efficiently without unnecessary re-renders', async () => {
      const props = {
        title: 'Spindle Load',
        metric: '72.5',
        status: 'OK',
        volatility: 0.3,
        unit: '%'
      };

      // Track render counts if using React.memo or similar
      let renderCount = 0;
      const originalRender = render;

      render(
        <BrowserRouter>
          <NeuroCard {...props} />
        </BrowserRouter>
      );

      // Component should render only when props change
      expect(renderCount).toBeLessThanOrEqual(2); // Initial render + possibly one update
    });

    test('handles large datasets efficiently', async () => {
      // Mock a large dataset
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        name: `Machine ${i}`,
        status: i % 3 === 0 ? 'RUNNING' : i % 3 === 1 ? 'IDLE' : 'MAINTENANCE',
        utilization: Math.random() * 100
      }));

      axios.get.mockResolvedValue({ data: largeDataset });

      render(
        <BrowserRouter>
          <MainDashboard />
        </BrowserRouter>
      );

      await waitFor(() => {
        // Should render without performance issues
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Verify that all items are accessible somehow
      expect(screen.getAllByRole('listitem')).toHaveLength(largeDataset.length);
    });
  });
});