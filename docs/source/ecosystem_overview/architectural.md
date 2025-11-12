# The Architectural Imperative: Performance Beyond Frameworks

As model architectures converge—for example, on multimodal Mixture-of-Experts (MoE) Transformers—the pursuit of peak performance is leading to the emergence of "Megakernels." A Megakernel is effectively the entire forward pass (or a large portion) of one specific model, hand-coded using a lower-level API like the CUDA SDK on NVIDIA GPUs. This approach achieves maximum hardware utilization by aggressively overlapping compute, memory, and communication. Recent work from the research community has demonstrated that this approach can yield significant throughput gains, over 22% in some cases, for inference on GPUs. This trend is not limited to inference; evidence suggests that some large-scale training efforts have involved low-level hardware control to achieve substantial efficiency gains.

If this trend accelerates, all high-level frameworks as they exist today risk becoming less relevant, as low-level access to the hardware is what ultimately matters for performance on mature, stable architectures. This presents a challenge for all modern ML stacks: how to provide expert-level hardware control without sacrificing the productivity and flexibility of a high-level framework.

For TPUs to provide a clear path to this level of performance, the ecosystem must expose an API layer that is closer to the hardware, enabling the development of these highly specialized kernels. As this report will detail, the JAX stack is designed to solve this by offering a continuum of abstraction (See Figure 2), from the automated, high-level optimizations of the XLA compiler to the fine-grained, manual control of the Pallas kernel-authoring library.

![](../_static/images/programming_TPUS.svg)

**Figure 2: The JAX continuum of abstraction**

