## The Core JAX AI Stack

The core JAX AI Stack consists of four key libraries that provide the foundation for model development: JAX, [Flax](https://flax.readthedocs.io/en/stable/), [Optax](https://optax.readthedocs.io/en/latest/), and [Orbax](https://orbax.readthedocs.io/en/latest/).

### **JAX: A Foundation for Composable, High-Performance Program Transformation** {#jax:-a-foundation-for-composable,-high-performance-program-transformation}

[JAX](https://docs.jax.dev/en/latest/) is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale Machine Learning. With its functional programming model and friendly, NumPy-like API, JAX provides a solid foundation for higher-level libraries.

With its compiler-first design, JAX inherently promotes scalability by leveraging [XLA](https://openxla.org/xla) (see Section \<\*\*\*\*\*\*\>) for aggressive, whole-program analysis, optimization, and hardware targeting. The JAX emphasis on functional programming (i.e., pure functions) makes its core program transformations more tractable and, crucially, composable

These core transformations can be mixed and matched to achieve high performance and scaling of workloads across model size, cluster size, and hardware types:

* **jit**: Just-in-time compilation of Python functions into optimized, fused XLA executables.  
* **grad**: Automatic differentiation, supporting forward- and reverse-mode, as well as higher-order derivatives.  
* **vmap**: Automatic vectorization, enabling seamless batching and data parallelism without modifying function logic.  
* **pmap / shard\_map**: Automatic parallelization across multiple devices (e.g., TPU cores), forming the basis for distributed training.

The seamless integration with XLA's GSPMD (General-purpose SPMD) model allows JAX to automatically parallelize computations across large TPU pods with minimal code changes. In most cases, scaling simply requires high-level sharding annotations, a stark contrast to frameworks where scaling may require more manual management of device placement and communication collectives

### **Flax: Flexible Neural Network Authoring and "Model Surgery"** {#flax:-flexible-neural-network-authoring-and-"model-surgery"}

#### **Flax \- neural network layers** {#flax---neural-network-layers}

[Flax](https://flax.readthedocs.io/en/latest/index.html) is a library designed to simplify the creation, debugging, and analysis of neural networks in JAX.  While pure functional API provided by JAX can be used to fully specify and train a ML model, users coming from the PyTorch (or TensorFlow) ecosystem are more used to and comfortable with the object oriented approach of specifying models as a graph of `torch.nn.Modules`. The abstractions provided by [Flax](https://flax.readthedocs.io/en/stable/) allow users to think more in terms of layers rather than functions, making it more developer friendly to an audience who value ergonomics and experimentation ease. [Flax](https://flax.readthedocs.io/en/stable/) also enables config driven model construction systems, such as those present in [MaxText](https://maxtext.readthedocs.io/en/latest/) and AxLearn, which separate out model hyperparameters from layer definition code.

With a simple Pythonic API, it allows developers to express models using regular Python objects, while retaining the power and performance of JAX. Flax's NNX API is an evolution of the Flax Linen interface, incorporating lessons learned to offer a more user-friendly interface that remains consistent with the core JAX APIs. Since Flax modules are fully backed by the core JAX APIs, there is no performance penalty associated with defining the model in [Flax](https://flax.readthedocs.io/en/stable/).

##### **Motivation** {#motivation}

JAX’s pure functional API, while powerful, can be complex for new users since it requires all the program state to be explicitly managed by the user. This paradigm can be unfamiliar to developers used to other frameworks. Modern model architectures are often complex with individual portions of the model trained separately and merged to form the final model[^3], in a process commonly referred to as model surgery. Even with decoder-only LLMs which tend to have a straightforward architecture, post training techniques such as LoRA and quantization require the model definition to be easily manipulated allowing parts of the architecture to be modified or even replaced.

The Flax NNX library, with its simple yet powerful Pythonic API enables this functionality in a way that is intuitive to the user, reducing the amount of cognitive overhead involved in authoring and training a model.

##### **Design** {#design}

The [Flax](https://flax.readthedocs.io/en/stable/) NNX library introduces an object oriented model definition system that encapsulates the model and random number generator state internally, reducing the cognitive overhead of the user and provides a familiar experience for those accustomed to frameworks like PyTorch or TensorFlow. By making submodule definitions Pythonic and providing APIs to traverse the module hierarchy, it allows for the model definition to be easily editable programmatically for model introspection and surgery.

The [Flax](https://flax.readthedocs.io/en/stable/) NNX APIs are designed to be consistent with the core JAX APIs to allow users to exploit the full expressibility and performance of JAX, with lifted transformations for common operations like sharding, jit and others. Models defined using the NNX APIs can also be adapted to work with functional training loops, allowing the user the flexibility they need while retaining an intuitive object oriented API.

##### **Key Strengths** {#key-strengths}

* **Intuitive object oriented flexible APIs:** Layers are represented as pure Python objects with internal state management, simplifying model construction and training loops, while also advanced model surgery use cases through support for submodule replacement, partial initialization and model hierarchy traversal.  
* **Consistent with Core JAX APIs:** Lifted transformations consistent with core JAX and fully compatible with functional JAX provide the full performance of JAX without sacrificing developer friendliness.


### **Optax: Composable Gradient Processing and Optimization Strategies** {#optax:-composable-gradient-processing-and-optimization-strategies}

[Optax](https://optax.readthedocs.io/en/latest/index.html) is a gradient processing and optimization library for JAX. It is designed to empower model builders  by providing building blocks that can be recombined in custom ways in order to train deep learning models amongst other applications. It builds on the capabilities of the core JAX library to provide a well tested high performance library of losses and optimizer functions and associated techniques that can be used to train ML models.

#### Motivation {#motivation-1}

The calculation and minimization of losses is at the core of what enables the training of ML models. With its support for automatic differentiation the core JAX library provides the numeric capabilities to train models, but it does not provide standard implementations of popular optimizers (ex. `RMSProp`, `Adam`)  or losses (`CrossEntropy`, `MSE` etc). While it is true that a user could implement these functions by themselves (and some advanced users will choose to do so), a bug in an optimizer implementation would introduce hard to diagnose model quality issues.  Rather than having the user implement such critical pieces, [Optax](https://optax.readthedocs.io/en/latest/) provides implementations of these algorithms that are tested for correctness and performance.

The field of optimization theory lies squarely in the realm of research, however its central role in training also makes it an indispensable part of training production ML models. A library that serves this role needs to be both flexible enough to accommodate rapid research iterations and also robust and performant enough to be dependable for production model training. It should also provide well tested implementations of state of the art algorithms which match the standard equations. The [Optax](https://optax.readthedocs.io/en/latest/) library, through its modular composable architecture and emphasis on correct readable code is designed to achieve this.

#### Design {#design-1}

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

#### Key Strengths {#key-strengths-1}

* **Robust Library:** Provides a comprehensive library of losses, optimizers, and algorithms with a focus on correctness and readability.  
* **Modular Chainable Transformations:** As shown above, this flexible API allows users to craft powerful, complex optimization strategies declaratively, without modifying the training loop.  
* **Functional and Scalable:** The pure functional implementations integrate seamlessly with JAX's parallelization mechanisms (e.g., pmap), enabling the same code to scale from a single host to large clusters.

### **Orbax / TensorStore \- Large scale distributed checkpointing** {#orbax-/-tensorstore---large-scale-distributed-checkpointing}

[**Orbax**](https://orbax.readthedocs.io/en/latest/) is an any-scale checkpointing library for JAX users backed primarily by [**TensorStore**](https://google.github.io/tensorstore/), a library for efficiently reading and writing multi-dimensional arrays. The two libraries operate at different levels of the stack \- Orbax at the level of ML models and states \- TensorStore at the level of individual arrays.

#### Motivation {#motivation-2}

[Orbax](https://orbax.readthedocs.io/en/latest/), which centers on JAX users and ML checkpointing, aims to reduce the fragmentation of checkpointing implementations across disparate research codebases, increase adoption of important performance features outside the most cutting-edge codebases, and provide a clean, flexible API for novice and advanced users alike. With advanced features like fully asynchronous distributed checkpointing, multi-tier checkpointing and emergency checkpointing, [Orbax](https://orbax.readthedocs.io/en/latest/) enables resilience in the largest of training jobs while also providing a flexible representation for publishing checkpoints.

#### ML Checkpointing vs Generalized Checkpoint/Restore {#ml-checkpointing-vs-generalized-checkpoint/restore}

It is worth considering the difference between ML checkpoint systems ([Orbax](https://orbax.readthedocs.io/en/latest/), NeMO-Megatron, Torch Distributed Checkpoint) with generalized checkpoint systems like CRIU. 

Systems like CRIU & CRIUgpu behave analogously to VM live migration; they halt the entire system and take a snapshot of every last bit of information so it can be faithfully reconstructed. This captures the entirety of the process’ host memory, device memory and operating system state. This is far more information that is actually needed to reconstruct a ML workload, since for a ML workload, a very large fraction of this information (activations, data examples, file handles) is trivially reconstructed. Capturing this much data also incurs a large amount of time when the job is halted.

ML checkpoint systems are designed to minimize the amount of time the accelerator is halted by selectively persisting information that cannot be reconstructed. Specifically, this entails persisting model weights, optimizer state, dataloader state and random number generator state, which is a far smaller amount of data.

#### Design {#design-2}

The [Orbax API](https://orbax.readthedocs.io/en/latest/index.html) centers around handling [PyTrees](https://docs.jax.dev/en/latest/pytrees.html) (nested containers) of arrays as the standard representation of JAX models. Saving and loading can be synchronous or asynchronous, with saving consisting of blocking and non-blocking phases. A higher-level `Checkpointer` class is provided, which facilitates checkpointing in a training loop, with save intervals, garbage collection, dataset checkpointing, and metadata management. Finally, Orbax provides customization layers for dealing with user-defined checkpointable objects and PyTree leaves.

The storage layer of [Orbax](https://orbax.readthedocs.io/en/latest/index.html) is the [TensorStore](https://google.github.io/tensorstore/) library, which is not technically part of the JAX ecosystem at all, and seeks to provide a flexible and highly versatile library for array storage. However, it is not designed around ML concepts and introduces too much complexity and manual management for most JAX users. [Orbax](https://orbax.readthedocs.io/en/latest/index.html) smooths out this experience to provide users an easy to use ML specific API surface.

To maximize the utilization of the accelerator, the checkpointing library must minimize the time it halts the training to snapshot the state. This is achieved by overlapping the checkpointing operations with the compute operations as shown in the diagram below. It’s worth noting that asynchronous checkpointing is table-stakes for large workloads and isn’t unique to [Orbax](https://orbax.readthedocs.io/en/latest/index.html). It is also present in other frameworks such as NeMO-Megatron and Torch Distributed Checkpoints.

When considering asynchronous checkpointing with non overlapped device-to-host transfers, the amount of time the accelerator is halted is thus a function of the number of model parameters, the size of the parameters and the PCI link speed. Enabling fully overlapped D2H can further reduce this time by overlapping the D2H transfer with the forward pass of the next step. As long as the D2H transfer can complete before the next forward step completes, the checkpoint will become effectively[^4] free.

Restarting from an error is similarly bound by two factors, the XLA compilation time and the speed of reading the weights back from storage. XLA compilation caches can make the former insignificant. Reading from storage is hardware dependent \- emergency checkpoints save to ramdisks which are extremely fast, however there is a speed spectrum that ranges from ramdisk to SSD, HDD and GCS.

Specific industry-leading performance features have their own design challenges, and merit separate attention:

* [**Async checkpointing**](https://orbax.readthedocs.io/en/latest/guides/checkpoint/async_checkpointing.html): Checkpointing only needs to block accelerator computations while data is being transferred from host to/from accelerator memory. Expensive I/O operations can take place in a background thread meaning save time can be reduced by 95-99% relative to blocking saves. Asynchronous loading is also possible, and can save time on startup, but requires more extensive effort to integrate and has not yet seen widespread adoption.  
* [**OCDBT format**](https://orbax.readthedocs.io/en/latest/guides/checkpoint/optimized_checkpointing.html): Most previous checkpointing implementations stored parameters as separate subdirectories, which caused significant overhead for small arrays. TensorStore’s OCDBT format uses an efficient [B+ tree](https://en.wikipedia.org/wiki/B%2B_tree) format, which allows fine-grained control over shard shapes and file sizes that can be tuned to different filesystems and models. The save/load strategy provides scalability to tens of thousands of nodes by ensuring each host independently reads and writes only the relevant pieces of each array.  
* [**Restore \+ broadcast**](https://cloud.google.com/blog/products/compute/unlock-faster-workload-start-time-using-orbax-on-jax): Hero-scale training runs replicate the model weights among multiple data-parallel replicas. Orbax provides a load balancing feature that distributes the burden evenly among available replicas when saving. It also leverages fast chip interconnects to avoid redundant reads of the model on different groups of hosts, instead loading on a single primary replica and broadcasting the weights to all other replicas.  
* **Emergency checkpointing**: Hero-scale training suffers from frequent interruptions and hardware failures. Checkpointing to persistent RAM disk improves goodput for hero-scale jobs by allowing for increased checkpoint frequency, faster restore times, and improved resiliency, since TPU states may be corrupted on some replicas, but not all.

#### Key Strengths {#key-strengths-2}

* **Widespread adoption:** As checkpoints are a medium for communication of ML artifacts between different codebases and stages of ML development, widespread adoption is an inherent advantage. Currently, Orbax has [\~4 million](https://pypistats.org/packages/orbax-checkpoint) monthly package downloads.   
* **Easy to use:** Orbax abstracts away complex technical aspects of checkpointing like async saving, single- vs. multi-controller, checkpoint atomicity, distributed filesystem details, TPU vs. GPU, etc. It condenses use cases into simple, but generalizable APIs (direct-to-path, sequence-of-steps).  
* **Flexible:** While Orbax focuses on exposing a simple API surface for the majority of users, additional layers for handling custom checkpointable objects and PyTree nodes allow for flexibility in specialized use cases.  
* **Performant and scalable:** Orbax provides a variety of features designed to make checkpointing as fast and as unobtrusive as possible, freeing developers to focus on efficiency in the remainder of the training loop. Scalability to the cutting edge of ML research is a top concern of the library; training runs at a scale of O(10k) nodes currently rely on Orbax.

#### **Grain: Deterministic and Scalable Input Data Pipelines** {#grain:-deterministic-and-scalable-input-data-pipelines}

[Grain](https://google-grain.readthedocs.io/en/latest/) is a Python library for reading and processing data for training and evaluating JAX models. It is flexible, fast and deterministic and supports advanced features like checkpointing which are essential to successfully training large workloads. It supports popular data formats and storage backends and also provides a flexible API to extend support to user specific formats and backends that are not natively supported. While [Grain](https://google-grain.readthedocs.io/en/latest/) is primarily designed to work with JAX, it is framework independent, does not require JAX to run and can be used with other frameworks as well.

##### **Motivation** {#motivation-7}

Data pipelines form a critical part of the training infrastructure \- they need to be flexible so that common transformations can be expressed efficiently, and performant enough that they are able to keep the accelerators busy at all times. They also need to be able to accommodate multiple storage formats and backends. Due to their higher step times, training large models at scale pose unique additional requirements on the data pipeline beyond those that are required by regular training workloads, primarily focused around determinism and reproducibility[^5]. The [Grain](https://google-grain.readthedocs.io/en/latest/) library is designed with a flexible enough architecture to address all these needs.

##### **Design** {#design-7}

At the highest level, there are two ways to structure an input pipeline, as a separate cluster of data workers or by co-locating the data workers on the hosts that drive the accelerators. [Grain](https://google-grain.readthedocs.io/en/latest/) chooses the latter for a variety of reasons.

Accelerators are combined with powerful hosts that typically sit idle during training steps, which makes it a natural choice to run the input data pipeline. There are however additional advantages to doing so \- it simplifies the user's view of data sharding by providing a consistent view of sharding across input and compute. It could be argued that putting the data worker on the accelerator host risks saturating the host CPU, however this does not preclude offloading compute intensive transformations to another cluster via RPCs[^6].

On the API front, with a pure python implementation that supports multiple processes and a flexible API, [Grain](https://google-grain.readthedocs.io/en/latest/) enables users to implement arbitrarily complex data transformations by composing together pipeline stages based on well understood [transformation](https://google-grain.readthedocs.io/en/latest/data_loader/transformations.html) paradigms.

Out of the box, [Grain](https://google-grain.readthedocs.io/en/latest/) supports efficient random access data formats like `ArrayRecord` and `Bagz` alongside other popular data formats such as Parquet and `TFDS`. [Grain](https://google-grain.readthedocs.io/en/latest/) includes support for reading from local file systems as well as reading from GCS by default. Along with supporting popular storage formats and backends, a clean abstraction to the storage layer allows users to easily add support for or wrap their existing data sources to be compatible with the [Grain](https://google-grain.readthedocs.io/en/latest/) library.

##### **Key Strengths** {#key-strengths-7}

* **Deterministic data feeding:** Colocating the data worker with the accelerator and coupling it with a stable global shuffle and [checkpointable iterators](https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html#checkpointing) allows the model state and data pipeline state to be checkpointed together in a consistent snapshot using [Orbax](https://docs.google.com/document/d/1rS4DGWSbHOX0rZgjv2rV2DcXuBnHvnCKOTAarZiC1Dg/edit?tab=t.0#heading=h.rtje6zr33hjw), enhancing the determinism of the training process.  
* **Flexible APIs to enable powerful data transformations:** A flexible pure Python [transformations](https://google-grain.readthedocs.io/en/latest/data_loader/transformations.html) API allows for extensive data transformations within the input processing pipeline.  
* **Extensible support for multiple formats and backends:** An extensible [data sources](https://google-grain.readthedocs.io/en/latest/tutorials/data_sources/index.html) API supports popular storage formats and backends and allows users to easily add support for new formats and backends.  
* **Powerful debugging interface:** Data pipeline [visualization tools](https://google-grain.readthedocs.io/en/latest/tutorials/dataset_debugging_tutorial.html) and a debug mode allow users to introspect, debug and optimize the performance of their data pipelines.

