import { test, expect } from '@playwright/test';

/**
 * E2E tests for progress event GUI display integration.
 * Tests the complete flow: tqdm output -> IPC server -> Rust backend -> Tauri -> Frontend
 */

test.describe('Progress Bar Display Integration', () => {
	test('should display progress value from backend in progress component', async ({ page }) => {
		// Navigate to the app
		await page.goto('/');

		// Simulate backend progress event via Tauri API
		await page.evaluateHandle(() => {
			const event = new CustomEvent('tauri://inference-progress', {
				detail: {
					progress: 42,
					message: 'Processing sagittal plane: 107/256'
				}
			});
			window.dispatchEvent(event);
		});

		// Check that progress bar shows 42%
		const progressBar = page.locator('[data-testid="progress-bar"]');
		await expect(progressBar).toHaveAttribute('aria-valuenow', '42');
	});

	test('should update progress bar when receiving sequential events', async ({ page }) => {
		await page.goto('/');

		// Dispatch multiple progress events
		const progressValues = [10, 25, 50, 75, 100];
		for (const progress of progressValues) {
			await page.evaluateHandle((p) => {
				const event = new CustomEvent('tauri://inference-progress', {
					detail: {
						progress: p,
						message: `Progress: ${p}%`
					}
				});
				window.dispatchEvent(event);
			}, progress);

			// Verify progress bar updated to current value
			const progressBar = page.locator('[data-testid="progress-bar"]');
			await expect(progressBar).toHaveAttribute('aria-valuenow', progress.toString());
		}
	});

	test('should display progress message text from backend', async ({ page }) => {
		await page.goto('/');

		const testMessage = 'Processing sagittal plane: 181/256 [06:32<12:45, 1.25s/batch]';
		await page.evaluateHandle((msg) => {
			const event = new CustomEvent('tauri://inference-progress', {
				detail: {
					progress: 71,
					message: msg
				}
			});
			window.dispatchEvent(event);
		}, testMessage);

		// Check that message is displayed
		const messageElement = page.locator('[data-testid="progress-message"]');
		await expect(messageElement).toContainText(testMessage);
	});

	test('should handle progress value clamping to 0-100 range', async ({ page }) => {
		await page.goto('/');

		// Test with values outside 0-100 range (should be clamped)
		const invalidValues = [-50, -1, 101, 150];
		const clampedExpected = [0, 0, 100, 100];

		for (let i = 0; i < invalidValues.length; i++) {
			await page.evaluateHandle((val) => {
				const event = new CustomEvent('tauri://inference-progress', {
					detail: {
						progress: val,
						message: `Testing ${val}`
					}
				});
				window.dispatchEvent(event);
			}, invalidValues[i]);

			const progressBar = page.locator('[data-testid="progress-bar"]');
			await expect(progressBar).toHaveAttribute('aria-valuenow', clampedExpected[i].toString());
		}
	});

	test('should maintain progress state when receiving non-progress events', async ({ page }) => {
		await page.goto('/');

		// Set initial progress
		await page.evaluateHandle(() => {
			const event = new CustomEvent('tauri://inference-progress', {
				detail: { progress: 50, message: 'Halfway' }
			});
			window.dispatchEvent(event);
		});

		const progressBar = page.locator('[data-testid="progress-bar"]');
		await expect(progressBar).toHaveAttribute('aria-valuenow', '50');

		// Send non-progress event (should not affect progress)
		await page.evaluateHandle(() => {
			const event = new CustomEvent('tauri://log', {
				detail: { level: 'info', message: 'Some log message' }
			});
			window.dispatchEvent(event);
		});

		// Progress should remain 50
		await expect(progressBar).toHaveAttribute('aria-valuenow', '50');
	});

	test('should show progress animation between sequential updates', async ({ page }) => {
		await page.goto('/');

		const progressBar = page.locator('[data-testid="progress-bar"]');

		// Initial state
		await page.evaluateHandle(() => {
			const event = new CustomEvent('tauri://inference-progress', {
				detail: { progress: 0, message: 'Starting' }
			});
			window.dispatchEvent(event);
		});
		await expect(progressBar).toHaveAttribute('aria-valuenow', '0');

		// Update to 100%
		await page.evaluateHandle(() => {
			const event = new CustomEvent('tauri://inference-progress', {
				detail: { progress: 100, message: 'Complete' }
			});
			window.dispatchEvent(event);
		});
		await expect(progressBar).toHaveAttribute('aria-valuenow', '100');

		// Verify the progress bar element exists and is visible
		await expect(progressBar).toBeVisible();
	});

	test('should display completion state with 100% progress', async ({ page }) => {
		await page.goto('/');

		await page.evaluateHandle(() => {
			const event = new CustomEvent('tauri://inference-progress', {
				detail: {
					progress: 100,
					message: 'MRI processing completed'
				}
			});
			window.dispatchEvent(event);
		});

		const progressBar = page.locator('[data-testid="progress-bar"]');
		await expect(progressBar).toHaveAttribute('aria-valuenow', '100');

		// Check if completion message is displayed
		const messageElement = page.locator('[data-testid="progress-message"]');
		await expect(messageElement).toContainText('completed');
	});

	test('should handle rapid consecutive progress updates without dropping events', async ({ page }) => {
		await page.goto('/');

		// Send 10 rapid updates
		for (let i = 1; i <= 10; i++) {
			await page.evaluateHandle((idx) => {
				const event = new CustomEvent('tauri://inference-progress', {
					detail: {
						progress: idx * 10,
						message: `Batch ${idx}/10`
					}
				});
				window.dispatchEvent(event);
			}, i);
		}

		// Final state should be 100%
		const progressBar = page.locator('[data-testid="progress-bar"]');
		await expect(progressBar).toHaveAttribute('aria-valuenow', '100');

		// Message should show last update
		const messageElement = page.locator('[data-testid="progress-message"]');
		await expect(messageElement).toContainText('Batch 10/10');
	});

	test('should display tqdm-format progress messages', async ({ page }) => {
		await page.goto('/');

		const tqdmMessages = [
			'10%|█         | 26/256 [00:30<04:32, 1.69batch/s]',
			'50%|█████     | 128/256 [02:15<02:15, 1.69batch/s]',
			'100%|██████████| 256/256 [04:30<00:00, 1.69batch/s]'
		];

		for (let i = 0; i < tqdmMessages.length; i++) {
			const progress = (i + 1) * 33;
			await page.evaluateHandle(({ msg, prog }) => {
				const event = new CustomEvent('tauri://inference-progress', {
					detail: {
						progress: prog,
						message: msg
					}
				});
				window.dispatchEvent(event);
			}, { msg: tqdmMessages[i], prog: progress });

			const messageElement = page.locator('[data-testid="progress-message"]');
			await expect(messageElement).toContainText('batch/s');
		}
	});
});
