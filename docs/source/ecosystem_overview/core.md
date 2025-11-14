# The Core JAX AI Stack

The core JAX AI Stack consists of five key libraries that provide the foundation for model development: JAX, [Flax](https://flax.readthedocs.io/en/stable/), [Optax](https://optax.readthedocs.io/en/latest/), [Orbax](https://orbax.readthedocs.io/en/latest/) and [Grain](https://google-grain.readthedocs.io/en/latest/).

## JAX: A Foundation for Composable, High-Performance Program Transformation

[JAX](https://docs.jax.dev/en/latest/) is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale Machine Learning. With its functional programming model and friendly, NumPy-like API, JAX provides a solid foundation for higher-level libraries.

With its compiler-first design, JAX inherently promotes scalability by leveraging [XLA](https://openxla.org/xla) (see the [XLA Section](#xla-section)) for aggressive, whole-program analysis, optimization, and hardware targeting. The JAX emphasis on functional programming (i.e., pure functions) makes its core program transformations more tractable and, crucially, composable

These core transformations can be mixed and matched to achieve high performance and scaling of workloads across model size, cluster size, and hardware types:

* **jit**: Just-in-time compilation of Python functions into optimized, fused XLA executables.
* **grad**: Automatic differentiation, supporting forward- and reverse-mode, as well as higher-order derivatives.
* **vmap**: Automatic vectorization, enabling seamless batching and data parallelism without modifying function logic.
* **pmap / shard\_map**: Automatic parallelization across multiple devices (e.g., TPU cores), forming the basis for distributed training.

The seamless integration with XLA's GSPMD (General-purpose SPMD) model allows JAX to automatically parallelize computations across large TPU pods with minimal code changes. In most cases, scaling simply requires high-level sharding annotations, a stark contrast to frameworks where scaling may require more manual management of device placement and communication collectives

## Flax: Flexible Neural Network Authoring and "Model Surgery"

[Flax](https://flax.readthedocs.io/en/latest/index.html) simplifies the creation, debugging, and analysis of neural networks in JAX by providing an intuitive, object-oriented approach to model building. While JAX's functional API is powerful, Flax offers a more familiar layer-based abstraction for developers accustomed to frameworks like PyTorch, without any performance penalty.

This design simplifies modern ML practices like "model surgery"—the process of modifying or combining trained model components. Techniques such as LoRA and quantization require easily manipulable model definitions, which Flax's NNX API provides through a simple, Pythonic interface. NNX encapsulates model state, reducing user cognitive load, and allows for programmatic traversal and modification of the model hierarchy.

### Key Strengths:

* Intuitive Object-Oriented API: Simplifies model construction and enables advanced use cases like submodule replacement and partial initialization.

* Consistent with Core JAX: Flax provides lifted transformations that are fully compatible with JAX's functional paradigm, offering the full performance of JAX with enhanced developer friendliness.



(optax:composable)=
## Optax: Composable Gradient Processing and Optimization Strategies

[Optax](https://optax.readthedocs.io/en/latest/index.html) is a gradient processing and optimization library for JAX. It is designed to empower model builders  by providing building blocks that can be recombined in custom ways in order to train deep learning models amongst other applications. It builds on the capabilities of the core JAX library to provide a well tested high performance library of losses and optimizer functions and associated techniques that can be used to train ML models.

### Motivation

The calculation and minimization of losses is at the core of what enables the training of ML models. With its support for automatic differentiation the core JAX library provides the numeric capabilities to train models, but it does not provide standard implementations of popular optimizers (ex. `RMSProp`, `Adam`)  or losses (`CrossEntropy`, `MSE` etc). While it is true that a user could implement these functions by themselves (and some advanced users will choose to do so), a bug in an optimizer implementation would introduce hard to diagnose model quality issues.  Rather than having the user implement such critical pieces, [Optax](https://optax.readthedocs.io/en/latest/) provides implementations of these algorithms that are tested for correctness and performance.

The field of optimization theory lies squarely in the realm of research, however its central role in training also makes it an indispensable part of training production ML models. A library that serves this role needs to be both flexible enough to accommodate rapid research iterations and also robust and performant enough to be dependable for production model training. It should also provide well tested implementations of state of the art algorithms which match the standard equations. The [Optax](https://optax.readthedocs.io/en/latest/) library, through its modular composable architecture and emphasis on correct readable code is designed to achieve this.

### Design

[Optax](https://optax.readthedocs.io/en/latest/) is designed to both enhance research velocity and the transition from research to production by providing readable, well-tested, and efficient implementations of core algorithms. Optax has uses beyond the context of deep learning, however in this context it can be viewed as a collection of well known loss functions, optimization algorithms and gradient transformations implemented in a pure functional fashion in line with the JAX philosophy. The collection of well known [losses](https://optax.readthedocs.io/en/latest/api/losses.html) and [optimizers](https://optax.readthedocs.io/en/latest/api/optimizers.html) enable users to get started with ease and confidence.

The modular approach taken by Optax easily allows users to [chain multiple optimizers](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html#chain) together followed by other common [transformations](https://optax.readthedocs.io/en/latest/api/transformations.html) like gradient clipping for example and [wrap](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html) it using common techniques like MultiStep or Lookahead to achieve powerful optimization strategies all within a few lines of code. The flexible interface allows for easy research into new optimization algorithms and also enables powerful second order optimization techniques like shampoo or muon.

```py
# Optax implementation of a RMSProp optimizer with a custom learning rate schedule, gradient clipping and gradient accumulation.
optimizer = optax.chain(
  optax.clip_by_global_norm(GRADIENT_CLIP_VALUE),
  optax.rmsprop(learning_rate=optax.cosine_decay_schedule(init_value=lr,decay_steps=decay)),
  optax.apply_every(k=ACCUMULATION_STEPS)
)

# The same thing, in PyTorch
optimizer = optim.RMSprop(model_params, lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
for i, (inputs, targets) in enumerate(data_loader):
    # ... Training loop body ...
    if (i + 1) % ACCUMULATION_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
        optimizer.step()
        scheduler.step()
 optimizer.zero_grad()
```

As it can be seen in the example above, setting up an optimizer with a custom learning rate, gradient clipping and gradient accumulation is a simple drop in replacement block of code, compared to PyTorch which forces the user to modify their training loop to directly manage the learning rate scheduler, gradient clipping and gradient accumulation.

### Key Strengths

* **Robust Library:** Provides a comprehensive library of losses, optimizers, and algorithms with a focus on correctness and readability.
* **Modular Chainable Transformations:** As shown above, this flexible API allows users to craft powerful, complex optimization strategies declaratively, without modifying the training loop.
* **Functional and Scalable:** The pure functional implementations integrate seamlessly with JAX's parallelization mechanisms (e.g., pmap), enabling the same code to scale from a single host to large clusters.


(orbax:tensorstore)=
## Orbax / TensorStore \- Large scale distributed checkpointing

[Orbax](https://orbax.readthedocs.io/en/latest/) is a checkpointing library for JAX designed for any scale, from single-device to large-scale distributed training. It aims to unify fragmented checkpointing implementations and deliver critical performance features, such as asynchronous and multi-tier checkpointing, to a wider audience. Orbax enables the resilience required for massive training jobs and provides a flexible format for publishing checkpoints.

Unlike generalized checkpoint/restore systems that snapshot the entire system state, ML checkpointing with Orbax selectively persists only the information essential for resuming training—model weights, optimizer state, and data loader state. This targeted approach minimizes accelerator downtime. Orbax achieves this by overlapping I/O operations with computation, a critical feature for large workloads. The time accelerators are halted is thus reduced to the duration of the device-to-host data transfer, which can be further overlapped with the next training step, making checkpointing nearly free from a performance perspective.
At its core, Orbax uses [TensorStore](https://google.github.io/tensorstore/) for efficient, parallel reading and writing of array data. The [Orbax API](https://orbax.readthedocs.io/en/latest/index.html) abstracts this complexity, offering a user-friendly interface for handling [PyTrees](https://docs.jax.dev/en/latest/pytrees.html), which are the standard representation of models in JAX.

### Key Strengths:

* [Widespread Adoption](https://orbax.readthedocs.io/en/latest/guides/checkpoint/async_checkpointing.html): With millions of monthly downloads, Orbax serves as a common medium for sharing ML artifacts.
* Easy to Use: Orbax abstracts away the complexities of distributed checkpointing, including asynchronous saving, atomicity, and filesystem details.
* Flexible: While offering simple APIs for common use cases, Orbax allows for customization to handle specialized requirements.
* Performant and Scalable: Features like asynchronous checkpointing, an efficient storage format ([OCDBT](https://orbax.readthedocs.io/en/latest/guides/checkpoint/optimized_checkpointing.html)), and intelligent data loading strategies ensure that Orbax scales to training runs involving tens of thousands of nodes.



## Grain: Deterministic and Scalable Input Data Pipelines

[Grain](https://google-grain.readthedocs.io/en/latest/) is a Python library for reading and processing data for training and evaluating JAX models. It is flexible, fast and deterministic and supports advanced features like checkpointing which are essential to successfully training large workloads. It supports popular data formats and storage backends and also provides a flexible API to extend support to user specific formats and backends that are not natively supported. While [Grain](https://google-grain.readthedocs.io/en/latest/) is primarily designed to work with JAX, it is framework independent, does not require JAX to run and can be used with other frameworks as well.

### Motivation

Data pipelines form a critical part of the training infrastructure \- they need to be flexible so that common transformations can be expressed efficiently, and performant enough that they are able to keep the accelerators busy at all times. They also need to be able to accommodate multiple storage formats and backends. Due to their higher step times, training large models at scale pose unique additional requirements on the data pipeline beyond those that are required by regular training workloads, primarily focused around determinism and reproducibility[^5]. The [Grain](https://google-grain.readthedocs.io/en/latest/) library is designed with a flexible enough architecture to address all these needs.

### Design

At the highest level, there are two ways to structure an input pipeline, as a separate cluster of data workers or by co-locating the data workers on the hosts that drive the accelerators. [Grain](https://google-grain.readthedocs.io/en/latest/) chooses the latter for a variety of reasons.

Accelerators are combined with powerful hosts that typically sit idle during training steps, which makes it a natural choice to run the input data pipeline. There are however additional advantages to doing so \- it simplifies the user's view of data sharding by providing a consistent view of sharding across input and compute. It could be argued that putting the data worker on the accelerator host risks saturating the host CPU, however this does not preclude offloading compute intensive transformations to another cluster via RPCs[^6].

On the API front, with a pure python implementation that supports multiple processes and a flexible API, [Grain](https://google-grain.readthedocs.io/en/latest/) enables users to implement arbitrarily complex data transformations by composing together pipeline stages based on well understood [transformation](https://google-grain.readthedocs.io/en/latest/data_loader/transformations.html) paradigms.

Out of the box, [Grain](https://google-grain.readthedocs.io/en/latest/) supports efficient random access data formats like `ArrayRecord` and `Bagz` alongside other popular data formats such as Parquet and `TFDS`. [Grain](https://google-grain.readthedocs.io/en/latest/) includes support for reading from local file systems as well as reading from GCS by default. Along with supporting popular storage formats and backends, a clean abstraction to the storage layer allows users to easily add support for or wrap their existing data sources to be compatible with the [Grain](https://google-grain.readthedocs.io/en/latest/) library.

### Key Strengths

* **Deterministic data feeding:** Colocating the data worker with the accelerator and coupling it with a stable global shuffle and [checkpointable iterators](https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html#checkpointing) allows the model state and data pipeline state to be checkpointed together in a consistent snapshot using [Orbax](https://orbax.readthedocs.io/en/latest/), enhancing the determinism of the training process.
* **Flexible APIs to enable powerful data transformations:** A flexible pure Python [transformations](https://google-grain.readthedocs.io/en/latest/data_loader/transformations.html) API allows for extensive data transformations within the input processing pipeline.
* **Extensible support for multiple formats and backends:** An extensible [data sources](https://google-grain.readthedocs.io/en/latest/tutorials/data_sources/index.html) API supports popular storage formats and backends and allows users to easily add support for new formats and backends.
* **Powerful debugging interface:** Data pipeline [visualization tools](https://google-grain.readthedocs.io/en/latest/tutorials/dataset_debugging_tutorial.html) and a debug mode allow users to introspect, debug and optimize the performance of their data pipelines.


[^5]:  In the Section 5.1 of the [Palm paper](https://dl.acm.org/doi/10.5555/3648699.3648939), the authors note that they observed very large loss spikes despite having gradient clipping enabled and the solution was to remove the offending data batches and restart training from a checkpoint before the loss spike. This is only possible with a fully deterministic and reproducible training setup.

[^6]:  This is indeed how multimodal data pipelines would need to operate \- image and audio tokenizers for example are models themselves which run in their own clusters on their own accelerators and the input pipelines would make RPCs out to convert data examples into streams of tokens.
