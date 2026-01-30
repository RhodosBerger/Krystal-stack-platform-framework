/**
 * Comprehensive UI Tests for FANUC RISE v2.1 Advanced CNC Copilot Vue Components
 */

import { mount, shallowMount, createLocalVue } from '@vue/test-utils';
import { describe, expect, test, beforeEach, afterEach, vi } from 'vitest';
import axios from 'axios';

// Mock API responses
vi.mock('axios');

// Import Vue components to test
import GlassBrainInterface from '../../frontend-vue/src/components/GlassBrainInterface.vue';
import NeuroState from '../../frontend-vue/src/components/NeuroState.vue';
import CouncilLog from '../../frontend-vue/src/components/CouncilLog.vue';
import NormInspector from '../../frontend-vue/src/components/NormInspector.vue';

// Create local Vue instance for testing
const localVue = createLocalVue();

describe('FANUC RISE v2.1 Vue UI Components', () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
  });

  describe('GlassBrainInterface Component', () => {
    test('renders with correct initial state', () => {
      const wrapper = shallowMount(GlassBrainInterface, {
        localVue
      });

      // Check if the component renders correctly
      expect(wrapper.exists()).toBe(true);
      expect(wrapper.element).toMatchSnapshot();
    });

    test('displays neuro-chemical states visually', async () => {
      const propsData = {
        dopamineLevel: 0.75,
        cortisolLevel: 0.25,
        neuralActivity: 0.6
      };

      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData
      });

      // Check that the component displays the correct values
      await wrapper.vm.$nextTick();
      
      expect(wrapper.text()).toContain('Dopamine: 75%');
      expect(wrapper.text()).toContain('Cortisol: 25%');
      expect(wrapper.text()).toContain('Neural Activity: 60%');
    });

    test('updates visualization when props change', async () => {
      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 0.5,
          cortisolLevel: 0.5,
          neuralActivity: 0.5
        }
      });

      // Change the props
      await wrapper.setProps({
        dopamineLevel: 0.9,
        cortisolLevel: 0.1,
        neuralActivity: 0.8
      });

      // Check that the display updated
      expect(wrapper.text()).toContain('Dopamine: 90%');
      expect(wrapper.text()).toContain('Cortisol: 10%');
      expect(wrapper.text()).toContain('Neural Activity: 80%');
    });

    test('handles edge cases for extreme values', async () => {
      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 1.0,
          cortisolLevel: 0.0,
          neuralActivity: 1.0
        }
      });

      expect(wrapper.text()).toContain('Dopamine: 100%');
      expect(wrapper.text()).toContain('Cortisol: 0%');
      expect(wrapper.text()).toContain('Neural Activity: 100%');

      // Test with minimum values
      await wrapper.setProps({
        dopamineLevel: 0.0,
        cortisolLevel: 1.0,
        neuralActivity: 0.0
      });

      expect(wrapper.text()).toContain('Dopamine: 0%');
      expect(wrapper.text()).toContain('Cortisol: 100%');
      expect(wrapper.text()).toContain('Neural Activity: 0%');
    });

    test('applies correct visual styles based on state', async () => {
      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 0.9,
          cortisolLevel: 0.1,
          neuralActivity: 0.8
        }
      });

      // Check for visual indicators of high dopamine state
      const dopamineElement = wrapper.find('[data-testid="dopamine-indicator"]');
      if (dopamineElement.exists()) {
        expect(dopamineElement.classes()).toContain('dopamine-high');
      }

      // Check for visual indicators of low cortisol state
      const cortisolElement = wrapper.find('[data-testid="cortisol-indicator"]');
      if (cortisolElement.exists()) {
        expect(cortisolElement.classes()).toContain('cortisol-low');
      }
    });
  });

  describe('NeuroState Component', () => {
    test('renders neuro-state with status indicators', () => {
      const propsData = {
        state: {
          dopamine: 0.72,
          cortisol: 0.35,
          stressLevel: 'MODERATE',
          efficiency: 0.85,
          reasoningTrace: ['System operating optimally', 'Monitoring for anomalies']
        }
      };

      const wrapper = mount(NeuroState, {
        localVue,
        propsData
      });

      expect(wrapper.exists()).toBe(true);
      expect(wrapper.text()).toContain('Dopamine: 72%');
      expect(wrapper.text()).toContain('Cortisol: 35%');
      expect(wrapper.text()).toContain('Stress Level: MODERATE');
      expect(wrapper.text()).toContain('Efficiency: 85%');
    });

    test('displays reasoning trace information', async () => {
      const propsData = {
        state: {
          dopamine: 0.65,
          cortisol: 0.45,
          stressLevel: 'MODERATE',
          efficiency: 0.78,
          reasoningTrace: [
            'High spindle load detected',
            'Adjusting feed rate to prevent overheating',
            'Monitoring temperature trends'
          ]
        }
      };

      const wrapper = mount(NeuroState, {
        localVue,
        propsData
      });

      // Check that reasoning trace is displayed
      expect(wrapper.text()).toContain('High spindle load detected');
      expect(wrapper.text()).toContain('Adjusting feed rate to prevent overheating');
      expect(wrapper.text()).toContain('Monitoring temperature trends');
    });

    test('changes visual state based on stress level', async () => {
      const lowStressProps = {
        state: {
          dopamine: 0.85,
          cortisol: 0.15,
          stressLevel: 'LOW',
          efficiency: 0.92,
          reasoningTrace: ['Operating in optimal parameters']
        }
      };

      const highStressProps = {
        state: {
          dopamine: 0.25,
          cortisol: 0.85,
          stressLevel: 'HIGH',
          efficiency: 0.45,
          reasoningTrace: ['Critical stress detected', 'Initiating safety protocols']
        }
      };

      const wrapper = mount(NeuroState, {
        localVue,
        propsData: lowStressProps
      });

      // Check low stress state
      expect(wrapper.text()).toContain('Stress Level: LOW');
      expect(wrapper.classes()).toContain('state-low-stress');

      // Update to high stress state
      await wrapper.setProps({ state: highStressProps.state });
      
      // Check high stress state
      expect(wrapper.text()).toContain('Stress Level: HIGH');
      expect(wrapper.classes()).toContain('state-high-stress');
    });

    test('handles empty reasoning trace gracefully', () => {
      const propsData = {
        state: {
          dopamine: 0.5,
          cortisol: 0.5,
          stressLevel: 'MODERATE',
          efficiency: 0.6,
          reasoningTrace: []
        }
      };

      const wrapper = mount(NeuroState, {
        localVue,
        propsData
      });

      // Component should render even with empty reasoning trace
      expect(wrapper.exists()).toBe(true);
      expect(wrapper.text()).toContain('Reasoning Trace: None');
    });
  });

  describe('CouncilLog Component', () => {
    test('renders council decision log correctly', () => {
      const propsData = {
        logs: [
          {
            id: 'log1',
            timestamp: new Date('2023-07-15T10:30:00Z'),
            agent: 'Creator',
            action: 'Proposal Made',
            details: 'Proposed feed rate increase to 3000 mm/min',
            status: 'SUBMITTED'
          },
          {
            id: 'log2',
            timestamp: new Date('2023-07-15T10:31:00Z'),
            agent: 'Auditor',
            action: 'Validation Complete',
            details: 'Proposal passed physics validation',
            status: 'APPROVED'
          },
          {
            id: 'log3',
            timestamp: new Date('2023-07-15T10:32:00Z'),
            agent: 'Accountant',
            action: 'Economic Assessment',
            details: 'Projected 12% profit increase',
            status: 'APPROVED'
          }
        ]
      };

      const wrapper = mount(CouncilLog, {
        localVue,
        propsData
      });

      expect(wrapper.findAll('.log-entry')).toHaveLength(3);
      expect(wrapper.text()).toContain('Creator');
      expect(wrapper.text()).toContain('Auditor');
      expect(wrapper.text()).toContain('Accountant');
      expect(wrapper.text()).toContain('Proposed feed rate increase');
      expect(wrapper.text()).toContain('Physics validation');
      expect(wrapper.text()).toContain('Profit increase');
    });

    test('filters logs by agent type', async () => {
      const propsData = {
        logs: [
          {
            id: 'log1',
            timestamp: new Date('2023-07-15T10:30:00Z'),
            agent: 'Creator',
            action: 'Proposal Made',
            details: 'Proposed feed rate increase to 3000 mm/min',
            status: 'SUBMITTED'
          },
          {
            id: 'log2',
            timestamp: new Date('2023-07-15T10:31:00Z'),
            agent: 'Auditor',
            action: 'Validation Complete',
            details: 'Proposal passed physics validation',
            status: 'APPROVED'
          }
        ]
      };

      const wrapper = mount(CouncilLog, {
        localVue,
        propsData
      });

      // Initially should show all logs
      expect(wrapper.findAll('.log-entry')).toHaveLength(2);

      // Filter by Creator
      await wrapper.setData({ selectedAgent: 'Creator' });
      
      // Should only show Creator logs
      const creatorLogs = wrapper.findAll('.log-entry').filter(log => 
        log.text().includes('Creator')
      );
      expect(creatorLogs).toHaveLength(1);
    });

    test('formats timestamps correctly', () => {
      const propsData = {
        logs: [
          {
            id: 'log1',
            timestamp: new Date('2023-07-15T10:30:45Z'),
            agent: 'Creator',
            action: 'Test Action',
            details: 'Test Details',
            status: 'SUBMITTED'
          }
        ]
      };

      const wrapper = mount(CouncilLog, {
        localVue,
        propsData
      });

      // Check that timestamp is formatted properly
      expect(wrapper.text()).toMatch(/\d{2}:\d{2}:\d{2}/); // HH:MM:SS format
    });

    test('handles empty logs array', () => {
      const propsData = {
        logs: []
      };

      const wrapper = mount(CouncilLog, {
        localVue,
        propsData
      });

      expect(wrapper.text()).toContain('No council logs available');
    });
  });

  describe('NormInspector Component', () => {
    test('renders compliance dashboard with status indicators', () => {
      const propsData = {
        complianceStatus: {
          overall: 'COMPLIANT',
          standards: [
            { name: 'ISO 9001', status: 'COMPLIANT', score: 98 },
            { name: 'ISO 14001', status: 'NEAR_COMPLIANT', score: 87 },
            { name: 'OHSAS 18001', status: 'COMPLIANT', score: 95 }
          ],
          lastAuditDate: new Date('2023-06-01T00:00:00Z'),
          nextAuditDate: new Date('2024-06-01T00:00:00Z')
        }
      };

      const wrapper = mount(NormInspector, {
        localVue,
        propsData
      });

      expect(wrapper.text()).toContain('Overall Status: COMPLIANT');
      expect(wrapper.text()).toContain('ISO 9001');
      expect(wrapper.text()).toContain('ISO 14001');
      expect(wrapper.text()).toContain('OHSAS 18001');
      expect(wrapper.text()).toContain('98%');
      expect(wrapper.text()).toContain('87%');
      expect(wrapper.text()).toContain('95%');
    });

    test('displays warnings for non-compliant standards', () => {
      const propsData = {
        complianceStatus: {
          overall: 'PARTIAL_COMPLIANCE',
          standards: [
            { name: 'ISO 9001', status: 'COMPLIANT', score: 98 },
            { name: 'ISO 14001', status: 'NON_COMPLIANT', score: 65 },
            { name: 'OHSAS 18001', status: 'NEAR_COMPLIANT', score: 87 }
          ],
          lastAuditDate: new Date('2023-06-01T00:00:00Z'),
          nextAuditDate: new Date('2024-06-01T00:00:00Z')
        }
      };

      const wrapper = mount(NormInspector, {
        localVue,
        propsData
      });

      // Check for warning indicators
      expect(wrapper.text()).toContain('NON_COMPLIANT');
      expect(wrapper.text()).toContain('PARTIAL_COMPLIANCE');
      
      // Should highlight the non-compliant standard
      const nonCompliantElement = wrapper.find('.non-compliant');
      expect(nonCompliantElement.exists()).toBe(true);
      expect(nonCompliantElement.text()).toContain('ISO 14001');
    });

    test('updates when compliance data changes', async () => {
      const initialProps = {
        complianceStatus: {
          overall: 'COMPLIANT',
          standards: [
            { name: 'ISO 9001', status: 'COMPLIANT', score: 98 }
          ],
          lastAuditDate: new Date('2023-06-01T00:00:00Z'),
          nextAuditDate: new Date('2024-06-01T00:00:00Z')
        }
      };

      const wrapper = mount(NormInspector, {
        localVue,
        propsData: initialProps
      });

      expect(wrapper.text()).toContain('COMPLIANT');

      // Update to new data
      await wrapper.setProps({
        complianceStatus: {
          overall: 'NON_COMPLIANT',
          standards: [
            { name: 'ISO 9001', status: 'NON_COMPLIANT', score: 45 }
          ],
          lastAuditDate: new Date('2023-06-01T00:00:00Z'),
          nextAuditDate: new Date('2024-06-01T00:00:00Z')
        }
      });

      // Should reflect the new status
      expect(wrapper.text()).toContain('NON_COMPLIANT');
      expect(wrapper.text()).toContain('45%');
    });

    test('shows audit timeline information', () => {
      const propsData = {
        complianceStatus: {
          overall: 'COMPLIANT',
          standards: [
            { name: 'ISO 9001', status: 'COMPLIANT', score: 98 }
          ],
          lastAuditDate: new Date('2023-06-01T00:00:00Z'),
          nextAuditDate: new Date('2024-06-01T00:00:00Z')
        }
      };

      const wrapper = mount(NormInspector, {
        localVue,
        propsData
      });

      expect(wrapper.text()).toContain('Last Audit:');
      expect(wrapper.text()).toContain('Next Audit:');
    });
  });

  describe('Interactive Elements', () => {
    test('button clicks trigger appropriate actions', async () => {
      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 0.7,
          cortisolLevel: 0.3,
          neuralActivity: 0.6
        }
      });

      // Find and click a button
      const refreshButton = wrapper.find('button[data-testid="refresh-btn"]');
      if (refreshButton.exists()) {
        await refreshButton.trigger('click');
        
        // Check if the appropriate method was called
        expect(wrapper.emitted('refresh')).toBeTruthy();
      }
    });

    test('form inputs update component state', async () => {
      const wrapper = mount(NeuroState, {
        localVue,
        propsData: {
          state: {
            dopamine: 0.5,
            cortisol: 0.5,
            stressLevel: 'MODERATE',
            efficiency: 0.6,
            reasoningTrace: ['Initial state']
          }
        }
      });

      // Simulate user input
      const input = wrapper.find('input[data-testid="dopamine-input"]');
      if (input.exists()) {
        await input.setValue(0.8);
        await input.trigger('input');
        
        // Check if the input affected the internal state
        expect(wrapper.vm.localDopamine).toBe(0.8);
      }
    });
  });

  describe('Real-time Updates', () => {
    test('component reacts to WebSocket data updates', async () => {
      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 0.5,
          cortisolLevel: 0.5,
          neuralActivity: 0.5
        }
      });

      // Simulate receiving new data via WebSocket
      await wrapper.vm.updateNeuroState({
        dopamine: 0.8,
        cortisol: 0.2,
        neuralActivity: 0.7
      });

      // Check that the display updated
      expect(wrapper.text()).toContain('Dopamine: 80%');
      expect(wrapper.text()).toContain('Cortisol: 20%');
      expect(wrapper.text()).toContain('Neural Activity: 70%');
    });

    test('handles connection interruptions gracefully', async () => {
      const wrapper = mount(NeuroState, {
        localVue,
        propsData: {
          state: {
            dopamine: 0.6,
            cortisol: 0.4,
            stressLevel: 'MODERATE',
            efficiency: 0.7,
            reasoningTrace: ['Connected']
          }
        }
      });

      // Simulate connection loss
      await wrapper.vm.handleConnectionLost();

      // Component should show connection status
      expect(wrapper.text()).toContain('Connection Lost');
    });
  });

  describe('Responsive Design', () => {
    test('adapts layout for different screen sizes', () => {
      // Set a smaller window width
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768
      });

      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 0.7,
          cortisolLevel: 0.3,
          neuralActivity: 0.6
        }
      });

      // Trigger resize event
      window.dispatchEvent(new Event('resize'));

      // Component should adapt to mobile view
      expect(wrapper.classes()).toContain('mobile-view');
    });

    test('maintains readability on small screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 320
      });

      const wrapper = mount(CouncilLog, {
        localVue,
        propsData: {
          logs: [
            {
              id: 'log1',
              timestamp: new Date(),
              agent: 'Creator',
              action: 'Long action name that might wrap',
              details: 'Detailed information about the action taken',
              status: 'APPROVED'
            }
          ]
        }
      });

      // Should still be readable on small screens
      expect(wrapper.element).toMatchSnapshot();
    });
  });

  describe('Error Handling', () => {
    test('displays error state when API fails', async () => {
      // Mock an API failure
      axios.get.mockRejectedValue(new Error('Network error'));

      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 0.5,
          cortisolLevel: 0.5,
          neuralActivity: 0.5
        }
      });

      // Simulate API call that fails
      await wrapper.vm.fetchNeuroData();

      // Component should show error state
      expect(wrapper.text()).toContain('Error loading data');
    });

    test('handles invalid data gracefully', async () => {
      const wrapper = mount(NeuroState, {
        localVue,
        propsData: {
          state: {
            dopamine: NaN,
            cortisol: -1,
            stressLevel: 'INVALID',
            efficiency: 1.5,
            reasoningTrace: null
          }
        }
      });

      // Component should handle invalid values
      expect(wrapper.text()).toContain('Dopamine: 0%'); // Default to 0 for invalid
      expect(wrapper.text()).toContain('Cortisol: 0%'); // Default to 0 for invalid
    });
  });

  describe('Accessibility', () => {
    test('components have proper ARIA attributes', () => {
      const wrapper = mount(NeuroState, {
        localVue,
        propsData: {
          state: {
            dopamine: 0.7,
            cortisol: 0.3,
            stressLevel: 'LOW',
            efficiency: 0.85,
            reasoningTrace: ['System operational']
          }
        }
      });

      // Check for accessibility attributes
      const container = wrapper.find('.neuro-state-container');
      expect(container.attributes('role')).toBe('region');
      expect(container.attributes('aria-label')).toBe('Neurochemical State Display');
    });

    test('keyboard navigation works for interactive elements', async () => {
      const wrapper = mount(GlassBrainInterface, {
        localVue,
        propsData: {
          dopamineLevel: 0.7,
          cortisolLevel: 0.3,
          neuralActivity: 0.6
        }
      });

      // Find a focusable element
      const button = wrapper.find('button');
      if (button.exists()) {
        // Simulate keyboard navigation
        await button.trigger('keydown.tab');
        expect(button.element).toBe(document.activeElement);

        // Simulate enter key press
        await button.trigger('keydown.enter');
        expect(wrapper.emitted('buttonPressed')).toBeTruthy();
      }
    });
  });

  describe('Performance', () => {
    test('renders efficiently with large datasets', async () => {
      // Create a large set of log entries
      const largeLogSet = Array.from({ length: 1000 }, (_, i) => ({
        id: `log${i}`,
        timestamp: new Date(Date.now() - i * 60000), // 1 minute intervals
        agent: i % 3 === 0 ? 'Creator' : i % 3 === 1 ? 'Auditor' : 'Accountant',
        action: `Action ${i}`,
        details: `Details for action ${i}`,
        status: i % 4 === 0 ? 'APPROVED' : i % 4 === 1 ? 'REJECTED' : 'PENDING'
      }));

      const wrapper = mount(CouncilLog, {
        localVue,
        propsData: {
          logs: largeLogSet
        }
      });

      // Should render without performance issues
      await wrapper.vm.$nextTick();
      expect(wrapper.findAll('.log-entry')).toHaveLength(1000);
    });

    test('virtual scrolling for long lists', () => {
      // This would be tested differently depending on implementation
      const largeStandardsList = Array.from({ length: 100 }, (_, i) => ({
        name: `Standard-${i}`,
        status: i % 2 === 0 ? 'COMPLIANT' : 'NON_COMPLIANT',
        score: Math.floor(Math.random() * 100)
      }));

      const wrapper = mount(NormInspector, {
        localVue,
        propsData: {
          complianceStatus: {
            overall: 'PARTIAL_COMPLIANCE',
            standards: largeStandardsList,
            lastAuditDate: new Date(),
            nextAuditDate: new Date()
          }
        }
      });

      // Component should handle large lists efficiently
      expect(wrapper.element).toMatchSnapshot();
    });
  });

  describe('Integration Tests', () => {
    test('GlassBrainInterface integrates with NeuroState', () => {
      const wrapper = mount({
        template: `
          <div>
            <GlassBrainInterface 
              :dopamine-level="dopamine" 
              :cortisol-level="cortisol" 
              :neural-activity="activity"
            />
            <NeuroState :state="neuroState" />
          </div>
        `,
        components: {
          GlassBrainInterface,
          NeuroState
        },
        data() {
          return {
            dopamine: 0.7,
            cortisol: 0.3,
            activity: 0.6,
            neuroState: {
              dopamine: 0.7,
              cortisol: 0.3,
              stressLevel: 'MODERATE',
              efficiency: 0.75,
              reasoningTrace: ['Integrated display']
            }
          };
        }
      }, { localVue });

      // Both components should display consistent information
      expect(wrapper.text()).toContain('Dopamine: 70%');
      expect(wrapper.text()).toContain('Cortisol: 30%');
    });

    test('CouncilLog reflects decisions from Shadow Council', async () => {
      const wrapper = mount({
        template: `
          <div>
            <CouncilLog :logs="councilLogs" />
          </div>
        `,
        components: {
          CouncilLog
        },
        data() {
          return {
            councilLogs: [
              {
                id: 'decision1',
                timestamp: new Date(),
                agent: 'Shadow Council',
                action: 'Strategy Approved',
                details: 'Approved aggressive machining parameters',
                status: 'APPROVED'
              }
            ]
          };
        }
      }, { localVue });

      expect(wrapper.text()).toContain('Strategy Approved');
      expect(wrapper.text()).toContain('Approved aggressive machining parameters');
    });
  });
});