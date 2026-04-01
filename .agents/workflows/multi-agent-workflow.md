---
description: Multi-agent workflow for feature development and bug fixing
---

# Multi-Agent Workflow
 
This workflow orchestrates the interaction between the Architect, Coder, Reviewer, and Tester agents to deliver high-quality features while strictly adhering to **SOLID principles**.

## Step 1: Planning (Architect)
- The Architect analyzes the task and existing codebase.
 - The Architect produces an `implementation_plan.md` artifact inside `app/specs/{feature_name}`.
- **Git Isolation**: The Architect sets up an isolated Git worktree for the feature development (e.g., in a `.worktrees/` directory) and ensures these paths are added to the root `.gitignore`.
 - **Output:** `implementation_plan.md` inside `app/specs/{feature_name}` (use lowercase, underscore-separated `feature_name`) and initialized worktree environment.

## Step 2: Human Approval (Architecture)
- **Action**: The USER reviews the `implementation_plan.md` inside `app/specs/{feature_name}`.
- **Constraint**: The workflow **must not proceed** until the user approves or directs adjustments.

## Step 3: Behaviour Driven Tests (Product Owner)
- The Product Owner analyzes the task/plan to define acceptance criteria.
- The Product Owner writes high-level BDD tests (Gherkin/features or similar) using **real dependencies**.
 - **Output:** New BDD requirements placed at `app/specs/{feature_name}/bdd_requirements.md` (one folder per feature; use lowercase, underscore-separated `feature_name`). BDD tests derived from these requirements must adhere to rules in `app/SKILLS.md`.

## Step 4: Human Approval (Tests)
- **Action**: The USER reviews the BDD tests to ensure they align with requirements.
- **Constraint**: The workflow **must not proceed** until the user confirms the test suite covers all necessary scenarios.

## Step 5: Code Implementation (Coder)
- The Coder implements the changes according to the approved plan.
- The Coder ensures all BDD tests from the Product Owner pass and follows the naming/placement conventions in `app/SKILLS.md`.
- Update documentation to reflect the changes.
- **Output:** Functional code changes using **real implementations only**.

## Step 6: Verification (Tester)
- The Tester performs additional unit/integration tests and edge-case verification.
- The Tester ensures all tests follow the `action_should_expected_behavior` naming convention and placement rules from `app/SKILLS.md`.
- The Tester ensures all tests follow the **Real Dependencies** and containerized policies.
- **Output:** `testing_summary.md` in test result folder which follows the placement conventions in `app/SKILLS.md`.

## Step 7: Final Review (Reviewer)
- The Reviewer examines the plans, BDD tests, code, and verification results.
- The Reviewer **triggers and verifies Quality Gates**:
    - Runs `ruff check .` for Python.
    - Runs `cargo clippy --manifest-path app/gui/workbench/src-tauri/Cargo.toml` for Rust.
    - Runs `tsc --noEmit --project app/gui/damadian-ui/tsconfig.json` for TypeScript.
- The Reviewer provides feedback and either approves or requests changes, strictly rejecting mocks and fakes.
- **Output:** Review results and quality gate report.

## Step 8: Finalization
- If all quality gates and tests pass, the task is considered complete.
- Documentation and final artifacts are updated as needed.
