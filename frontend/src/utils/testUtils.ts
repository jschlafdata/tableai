// src/utils/testUtils.ts

/**
 * Test ID Generation Utilities
 * Minimal utilities for generating consistent test IDs for Python-based testing
 */

// Base types for test ID components
export type TestIdPrefix = 
  | "btn"      // Buttons
  | "card"     // Card containers
  | "status"   // Status indicators
  | "msg"      // Messages/alerts
  | "input"    // Input fields
  | "link"     // Links
  | "modal"    // Modals
  | "dropdown" // Dropdowns
  | "form"     // Forms
  | "table"    // Tables
  | "row"      // Table rows
  | "header"   // Headers
  | "nav"      // Navigation
  | "list"     // Lists
  | "section"  // Page sections
  | "icon"     // Icons
  | "label"    // Labels
  | "error"    // Error messages
  | "loading"; // Loading indicators

// Common actions
export type TestAction = 
  | "connect"
  | "disconnect"
  | "sync"
  | "reauth"
  | "settings"
  | "submit"
  | "cancel"
  | "save"
  | "delete"
  | "edit"
  | "create"
  | "refresh"
  | "download"
  | "upload";

// Integration types
export type IntegrationType = 
  | "dropbox"
  | "googleDrive"
  | "onedrive"
  | "slack"
  | "github";

// Status types
export type StatusType = 
  | "connected"
  | "disconnected"
  | "pending"
  | "error"
  | "warning"
  | "success"
  | "needs_reauth"
  | "syncing"
  | "loading";

// Configuration for generating test IDs
export interface TestIdConfig {
  prefix: TestIdPrefix;
  feature?: string;                        // Feature or module name (e.g., "integrations")
  integration?: IntegrationType | string;  // Integration name
  action?: TestAction;                     // Action being performed
  status?: StatusType;                     // Current status
  identifier?: string | number;            // Unique identifier (e.g., index, ID)
  suffix?: string;                         // Additional suffix
}

// Test attributes to spread onto elements
export interface TestAttributes {
  id?: string;
  "data-testid"?: string;
  "data-feature"?: string;
  "data-integration"?: string;
  "data-action"?: TestAction;
  "data-status"?: StatusType;
  "aria-label"?: string;
}

/**
 * Generates a consistent test ID from configuration
 * Format: prefix[-feature][-integration][-action][-identifier][-status][-suffix]
 * 
 * Examples:
 * - btn-integrations-dropbox-connect
 * - status-dropbox-connected
 * - msg-integrations-success
 */
export const generateTestId = (config: TestIdConfig): string => {
  const parts: (string | number)[] = [config.prefix];
  
  if (config.feature) parts.push(config.feature);
  if (config.integration) parts.push(config.integration);
  if (config.action) parts.push(config.action);
  if (config.identifier !== undefined) parts.push(config.identifier);
  if (config.status) parts.push(config.status);
  if (config.suffix) parts.push(config.suffix);
  
  return parts.join("-");
};

/**
 * Generates test attributes to spread onto React elements
 * 
 * Usage:
 * <button {...getTestAttributes({ 
 *   prefix: "btn",
 *   feature: "integrations", 
 *   integration: "dropbox",
 *   action: "connect"
 * })}>
 *   Connect
 * </button>
 * 
 * Output attributes:
 * - id="btn-integrations-dropbox-connect"
 * - data-testid="btn-integrations-dropbox-connect"
 * - data-feature="integrations"
 * - data-integration="dropbox"
 * - data-action="connect"
 * - aria-label="connect dropbox"
 */
export const getTestAttributes = (
  config: TestIdConfig & { ariaLabel?: string }
): TestAttributes => {
  const testId = generateTestId(config);
  
  const attributes: TestAttributes = {
    id: testId,
    "data-testid": testId,
  };

  // Add data attributes for easier Python/Selenium querying
  if (config.feature) attributes["data-feature"] = config.feature;
  if (config.integration) attributes["data-integration"] = config.integration;
  if (config.action) attributes["data-action"] = config.action;
  if (config.status) attributes["data-status"] = config.status;
  
  // Add aria-label for accessibility (auto-generate if not provided)
  if (config.ariaLabel) {
    attributes["aria-label"] = config.ariaLabel;
  } else if (config.action && (config.integration || config.feature)) {
    attributes["aria-label"] = `${config.action} ${config.integration || config.feature}`.trim();
  }
  
  return attributes;
};