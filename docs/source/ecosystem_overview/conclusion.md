## Conclusion: A Durable, Production-Ready Platform for the Future of AI 

The data provided in the table above draws to a rather simple conclusion \- these stacks have their own strengths and weaknesses in a small number of areas but overall are vastly similar from the software standpoint. Both stacks provide out of the box turnkey solutions for pre-training, post-training adaptation and deployment of foundational models.

The JAX AI stack offers a compelling and robust solution for training and deploying ML models at any scale. It leverages deep vertical integration across software and TPU hardware to deliver class-leading performance and total cost of ownership.

By building on battle-tested internal systems, the stack has evolved to provide inherent reliability and scalability, enabling users to confidently develop and deploy even the largest models. Its modular and composable design, rooted in the JAX ecosystem philosophy, grants users unparalleled freedom and control, allowing them to tailor the stack to their specific needs without the constraints of a monolithic framework.

With XLA and Pathways providing a scalable and fault-tolerant base, JAX providing a performant and expressive numerics library, powerful core development libraries like [Flax](https://flax.readthedocs.io/en/stable/), Optax, [Grain](https://google-grain.readthedocs.io/en/latest/), and [Orbax](#orbax-/-tensorstore---large-scale-distributed-checkpointing), advanced performance tools like Pallas, Tokamax, and Qwix, and a robust application and production layer in [MaxText](#foundation-model-training:-maxtext-and-maxdiffusion), vLLM, and NSE, the JAX AI stack provides a durable foundation for users to build on and rapidly bring state-of-the-art research to production.

[^1]:  Included in the [jax-ai-stack Python package](https://docs.jaxstack.ai/en/latest/install.html)

[^2]:  Included in the [jax-ai-stack Python package](https://docs.jaxstack.ai/en/latest/install.html)

[^3]:  Image diffusion models are a typical example of this and can commonly be divided logically into a separately trained prompt encoder and a diffusion backbone.

[^4]:  We say effectively free since there could be other bottlenecks such as the DMA engines, HBM bandwidth contention etc. that still incur a performance penalty.

[^5]:  In the Section 5.1 of the [Palm paper](https://dl.acm.org/doi/10.5555/3648699.3648939), the authors note that they observed very large loss spikes despite having gradient clipping enabled and the solution was to remove the offending data batches and restart training from a checkpoint before the loss spike. This is only possible with a fully deterministic and reproducible training setup. 

[^6]:  This is indeed how multimodal data pipelines would need to operate \- image and audio tokenizers for example are models themselves which run in their own clusters on their own accelerators and the input pipelines would make RPCs out to convert data examples into streams of tokens.

[^7]:  This is a well established paradigm and has precedent in the CPU world, where compiled code forms the bulk of the program with developers dropping down to intrinsics or inline assembly to optimize performance critical sections.

[^8]:  Some of the equivalents here are not true 1:1 comparisons because other frameworks draw API boundaries differently compared to JAX. The list of equivalents is not exhaustive and there are new libraries appearing frequently.

