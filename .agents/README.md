# Agents Configuration

This directory contains the configuration and workflow definitions for the multi-agent system.

## Strict Execution Policy
- **MANDATORY**: Agents are **ONLY** allowed to access, read, or modify files within the **current working directory** (repository root and subdirectories).
- **PROHIBITED**: Accessing files outside the project root (e.g., `/home/user/`, `/etc/`, `/tmp/` etc.) is strictly forbidden. 
- **Enforcement**: Any plan, code, or test attempting to reference absolute paths outside the workspace must be rejected by the Reviewer.

## Directory Structure

- `roles/`: Detailed descriptions of each agent's responsibilities and interaction patterns.
- `workflows/`: Orchestration flows that define how agents work together on tasks.

## Agents

- **Architect**: Designs systems and creates implementation plans.
- **Product Owner**: Defines acceptance criteria and writes BDD tests.
- **Coder**: Implements code changes based on plans.
- **Tester**: Ensures correctness through automated tests and verification.
- **Reviewer**: Reviews code for quality and standards.

## Usage

To start a new multi-agent task, refer to the [multi-agent-workflow](workflows/multi-agent-workflow.md), which now follows an 8-step gated process including mandatory human approvals and quality gates.
