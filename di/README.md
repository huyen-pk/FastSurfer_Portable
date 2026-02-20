# Dependency Injection in FastSurferCNN

## Overview

This document describes the dependency injection (DI) refactoring implemented in the FastSurferCNN module using the `injector` framework. The refactoring improves code testability, modularity, and maintainability by decoupling component creation from component usage and managing object lifecycles automatically.

## What is Dependency Injection?

Dependency Injection is a design pattern where objects receive their dependencies from external sources rather than creating them internally. This allows for:

- **Better testability**: Dependencies can be easily mocked or stubbed in tests
- **Increased modularity**: Components can be replaced without modifying dependent code
- **Improved maintainability**: Changes to dependency creation are isolated
- **Enhanced flexibility**: Different implementations can be swapped at runtime
- **Lifecycle management**: The injector framework manages object lifecycles (singleton, per-request, etc.)

## DI Framework

FastSurferCNN uses the [`injector`](https://github.com/python-injector/injector) framework, a lightweight dependency injection framework for Python. This provides:

- **Automatic dependency resolution**: The framework automatically instantiates and injects dependencies
- **Lifecycle management**: Control whether instances are singletons or created per-request
- **Type-safe**: Uses Python type hints for dependency resolution
- **No manual wiring**: Components are truly injected, not manually instantiated

## DI Components

### Location

All DI components are located in the `FastSurferCNN/di/` directory:

```
FastSurferCNN/di/
├── __init__.py
├── config_loader.py
├── container.py         # NEW: Injector configuration
├── device_manager.py
└── factories.py
```

## Using the Injector

### Basic Usage

```python
from FastSurferCNN.di import create_injector, DeviceManager, ModelFactory

# Create the injector
injector = create_injector(cfg=your_config)

# Get dependencies - they are automatically instantiated
device_manager = injector.get(DeviceManager)
model_factory = injector.get(ModelFactory)
```

### With Trainer

```python
from FastSurferCNN.train import Trainer
from FastSurferCNN.di import create_injector

# Create injector with configuration
injector = create_injector(cfg=cfg)

# Pass injector to Trainer - it will inject all dependencies
trainer = Trainer(cfg, injector=injector)

# Or use default (backward compatible)
trainer = Trainer(cfg)  # Creates its own injector internally
```

### With Inference

```python
from FastSurferCNN.inference import Inference
from FastSurferCNN.di import create_injector

# Create injector
injector = create_injector(cfg=cfg, device=device)

# Pass to Inference
inference = Inference(cfg, device, injector=injector)

# Or use default (backward compatible)
inference = Inference(cfg, device)  # Creates its own injector internally
```

## Component Details

### DeviceManager

The `DeviceManager` class provides centralized device allocation for PyTorch models.

**Lifecycle**: Singleton (one instance per injector)

**Features:**
- Automatic device selection (CUDA if available, otherwise CPU)
- Explicit device specification
- Parallel computation capability detection

**Usage:**

```python
# Via injector (recommended)
injector = create_injector()
device_manager = injector.get(DeviceManager)
device = device_manager.get_device()

# Check parallel capability
if device_manager.is_parallel_capable():
    # Use DataParallel
    pass
```

### ConfigLoader

The `ConfigLoader` class provides centralized configuration loading and management.

**Features:**
- Load configuration from YAML files
- Merge configuration from command-line arguments
- Programmatic configuration creation
- Flexible configuration override

**Usage:**

```python
from FastSurferCNN.di import ConfigLoader
import argparse

# Load from arguments
loader = ConfigLoader()
args = argparse.Namespace(cfg_file="config.yaml", opts=None)
cfg = loader.load_config(args)

# Load from file only
cfg = loader.load_from_file("config.yaml")

# Create with overrides
cfg = loader.create_config(
    cfg_file="config.yaml",
    opts=["MODEL.NUM_CLASSES", "50"],
    rng_seed=42
)
```

### Factory Classes

Factory classes encapsulate the creation logic for various components:

#### ModelFactory

Creates neural network models based on configuration.

```python
from FastSurferCNN.di import ModelFactory

factory = ModelFactory()
model = factory.create_model(cfg)
```

#### LossFunctionFactory

Creates loss functions based on configuration.

```python
from FastSurferCNN.di import LossFunctionFactory

factory = LossFunctionFactory()
loss_func = factory.create_loss_function(cfg)
```

#### OptimizerFactory

Creates optimizers based on configuration.

```python
from FastSurferCNN.di import OptimizerFactory

factory = OptimizerFactory()
optimizer = factory.create_optimizer(model, cfg)
```

#### DataLoaderFactory

Creates data loaders based on configuration.

```python
from FastSurferCNN.di import DataLoaderFactory

factory = DataLoaderFactory()
train_loader = factory.create_dataloader(cfg, "train")
val_loader = factory.create_dataloader(cfg, "val")
```

#### SchedulerFactory

Creates learning rate schedulers based on configuration.

```python
from FastSurferCNN.di import SchedulerFactory

factory = SchedulerFactory()
scheduler = factory.create_scheduler(optimizer, cfg)
```

## Refactored Classes

### Trainer Class

The `Trainer` class has been updated to accept dependencies via constructor injection.

**Before:**

```python
trainer = Trainer(cfg)
# Dependencies were created internally
```

**After:**

```python
# Default usage (backward compatible)
trainer = Trainer(cfg)

# Custom usage with dependency injection
device_manager = DeviceManager()
model_factory = ModelFactory()
loss_factory = LossFunctionFactory()

trainer = Trainer(
    cfg,
    device_manager=device_manager,
    model_factory=model_factory,
    loss_factory=loss_factory
)
```

### Inference Class

The `Inference` class has been updated to accept dependencies via constructor injection.

**Before:**

```python
inference = Inference(cfg, device)
# Dependencies were created internally
```

**After:**

```python
# Default usage (backward compatible)
inference = Inference(cfg, device)

# Custom usage with dependency injection
device_manager = DeviceManager(device)
model_factory = ModelFactory()

inference = Inference(
    cfg,
    device,
    device_manager=device_manager,
    model_factory=model_factory
)
```

## Testing

Unit tests for DI components are located in `test/unit/`:

- `test_device_manager.py`: Tests for DeviceManager
- `test_factories.py`: Tests for all factory classes

Run tests with:

```bash
python -m pytest test/unit/ -v
```

## Benefits of This Refactoring

1. **Backward Compatibility**: All existing code continues to work without modification. The DI parameters are optional and default to the original behavior.

2. **Improved Testability**: Components can now be tested in isolation by injecting mock dependencies.

3. **Flexibility**: Different implementations can be easily swapped without modifying the core classes.

4. **Separation of Concerns**: Component creation logic is separated from business logic.

5. **Easier Debugging**: Device management and model creation are now explicitly visible in the constructor.

## Migration Guide

### For Existing Code

No changes are required! All existing code will continue to work:

```python
# This still works exactly as before
trainer = Trainer(cfg)
inference = Inference(cfg, device)
```

### For New Code

Take advantage of dependency injection for better testability:

```python
# Testing example
def test_trainer_with_mock_model():
    cfg = mock_config()
    
    # Create mock factories
    mock_model_factory = MagicMock()
    mock_model_factory.create_model.return_value = mock_model
    
    # Inject dependencies
    trainer = Trainer(cfg, model_factory=mock_model_factory)
    
    # Test trainer behavior
    assert trainer.model == mock_model
```

## Future Enhancements

Potential areas for further improvement:

1. Create abstract base classes for factories to enable interface-based programming
2. Implement a DI container for managing complex dependency graphs
3. Extend DI to other modules (CerebNet, HypVINN, etc.)
4. Add configuration validation at the factory level
5. Implement lazy initialization for expensive resources

## References

- [Dependency Injection Pattern](https://en.wikipedia.org/wiki/Dependency_injection)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/best_practices.html)
