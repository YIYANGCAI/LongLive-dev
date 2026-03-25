# Research Project: [$A^4$ Cus: Adding Anyone at Anytime in Autoregressive Video Generation]

## Base Repository
- **Repo**: [LongLive-dev](https://github.com/YIYANGCAI/LongLive-dev)
- **Paper**: [LongLive: Real-time Interactive Long Video Generation, Yang Shuai, ICLR 2026, [PDF](https://arxiv.org/abs/2509.22622)]
- **What it does**: It introduces an interactive prompt mechanism in autoregressive video generation. This allows the video model to generate the initial clips based on the first prompt, and then further adapt to new prompts provided by the user, generating subsequent clips accordingly.

---

## Algorithm Overview (My Understanding)

This paper presents a frame-level AR framework for real-time interactive long video generation. The paper mainly focuses on efficiency design. The authors introduce a KV-recache mechanism that refreshes cached states to ensure smooth, adherent transitions when prompts change. They also propose a tuning strategy to address quality degradation in long videos, aligning training with inference. For efficiency, the model combines short window attention with a frame sink to maintain long-range consistency while accelerating generation. The resulting 1.3B parameter model achieves 20.7 FPS on an H100 GPU, supports videos up to 240 seconds, and demonstrates strong performance on VBench for both short and long video benchmarks.

---

## Codebase Map
The training process of long videos calls following python files 

distillation.py: Trainer.fwdbwd_one_step_streaming  # single training step.     

↓ 

streaming_training.py: StreamingTrainingModel.generate_next_chunk  # generate a chunk of video frames    

↓ 

streaming_training.py: StreamingTrainingModel._generate_chunk  # core generation function, including prompt switching determination

↓ 

streaming_switch_training.py: StreamingSwitchTrainingPipeline.generate_chunk_with_cache  # the basic denoising process (Run switch prompt)

---

## My Research Ideas

### Overview
Since LongLive only introduces prompts as control signals, I would like to explore another type of control signal: reference images. By providing the video generation model with a reference image that contains a specific character or object, the model can generate video content featuring that character or object.

In interactive video generation, a reference image can be input to the model in a similar way as a prompt. The subsequently generated video will then contain the person or object from that image.

### Idea 1: ID Memory Bank
**Motivation**: When a new reference image (subject) is added to the model, it should interact properly with previous references. Therefore, I propose adding a module called ID Memory Bank, which dynamically updates the references to be incorporated into the video being generated.

**Basic Implementation**: The common approach for adding a reference image is to feed it into the video model's 3D-VAE, convert it into latent tokens, and concatenate them with the video tokens. Through self-attention and cross-attention, the identity information from the reference image can be injected into the video tokens, enabling the model to generate video containing that identity.

## Instruction
+ Read the source code following the logic outlined in the Codebase Map.
+ Analyze how to integrate reference image control based on the information provided in Idea 1.
+ Note: No coding is required at this stage.