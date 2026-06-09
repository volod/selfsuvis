# nanochat training report

Generated: 2026-06-08 12:11:28

## Environment

### Git Information
- Branch: main
- Commit: 5124e88 (clean)
- Message: SV-28 | fixed times logging

### Hardware
- Platform: Linux
- CPUs: 20 cores (20 logical)
- Memory: 62.2 GB
- GPUs: 1x NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU
- GPU Memory: 11.5 GB total
- CUDA Version: 12.8
- Hourly Rate: $2.00/hour

### Software
- Python: 3.12.3
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 525,455
- Lines: 12,482
- Files: 51
- Tokens (approx): 131,363
- Dependencies (uv.lock lines): 4,897

Run started: 2026-06-08 12:11:28

---

## Tokenizer training
timestamp: 2026-06-08 12:12:07

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 36.7752
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.5827
- token_bytes_std: 2.8123


## Tokenizer evaluation
timestamp: 2026-06-08 12:12:09

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 405 | 4.49 | -0.2% |
| korean | 893 | 745 | 1.20 | 749 | 1.19 | -0.5% |
| code | 1259 | 576 | 2.19 | 397 | 3.17 | +31.1% |
| math | 1834 | 936 | 1.96 | 911 | 2.01 | +2.7% |
| science | 1112 | 260 | 4.28 | 247 | 4.50 | +5.0% |
| fwe-train | 2948778 | 631304 | 4.67 | 622480 | 4.74 | +1.4% |
| fwe-val | 3024593 | 653067 | 4.63 | 644914 | 4.69 | +1.2% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 405 | 4.49 | -4.7% |
| korean | 893 | 364 | 2.45 | 749 | 1.19 | -105.8% |
| code | 1259 | 309 | 4.07 | 397 | 3.17 | -28.5% |
| math | 1834 | 832 | 2.20 | 911 | 2.01 | -9.5% |
| science | 1112 | 249 | 4.47 | 247 | 4.50 | +0.8% |
| fwe-train | 2948778 | 611619 | 4.82 | 622480 | 4.74 | -1.8% |
| fwe-val | 3024593 | 631183 | 4.79 | 644914 | 4.69 | -2.2% |


## Base model training
timestamp: 2026-06-08 19:36:42

- run: 
- device_type: cuda
- fp8: False
- fp8_recipe: tensorwise
- depth: 12
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 1024
- window_pattern: SSSL
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 12.0000
- device_batch_size: 4
- total_batch_size: 524,288
- embedding_lr: 0.3000
- unembedding_lr: 0.0080
- weight_decay: 0.2800
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- resume_from_step: -1
- tokenizer_threads: 8
- tokenizer_batch_size: 128
- loader_buffer_size: 4000
- eval_every: 250
- eval_tokens: 5,242,880
- eval_at_start: False
- core_metric_every: 2000
- core_metric_max_per_task: 200
- sample_every: 500
- save_every: 1000
- model_tag: None
- Number of parameters: 286,261,730
- Number of FLOPs per token: 7.101507e+08
- Calculated number of iterations: 2520
- Number of training tokens: 1,321,205,760
- Tokens : Scaling params ratio: 12.0000
- DDP world size: 1
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- Minimum validation bpb: 0.8627
- Final validation bpb: 0.8627
- CORE metric estimate: 0.1520
- MFU %: 49.70%
- Total training flops: 9.382552e+17
- Total training time: 442.75m
- Peak memory usage: 6303.73MiB


## Base model evaluation
timestamp: 2026-06-08 19:38:02

- model: base_model (step 2520)
- CORE metric: 0.1520
- train bpb: 0.8451
- val bpb: 0.8360
- hellaswag_zeroshot: 0.1800
- jeopardy: 0.0050
- bigbench_qa_wikidata: 0.3200
- arc_easy: 0.3933
- arc_challenge: 0.0800
- copa: 0.1200
- commonsense_qa: 0.1313
- piqa: 0.4400
- openbook_qa: 0.1267
- lambada_openai: 0.3500
- hellaswag: 0.1600
- winograd: 0.1800
- winogrande: -0.0400
- bigbench_dyck_languages: 0.0450
- agi_eval_lsat_ar: 0.0062
- bigbench_cs_algorithms: 0.3750
- bigbench_operators: 0.1250
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.1550
- coqa: 0.1300
- boolq: -0.0921
- bigbench_language_identification: 0.1529
- sample 0: <|bos|>The capital of France is Paris, and the capital of the French Republic is Paris. The capital of France
- sample 1: <|bos|>The chemical symbol of gold is Au. It is a soft, malleable, and ductile metal that is
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Friday. The day of the week is Friday. The day of the month is
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is hot. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Jupiter, Saturn, Uranus, Neptune, and Pluto. The planets of
- sample 5: <|bos|>My favorite color is red. I love red because it's so powerful. I love red because it
- sample 6: <|bos|>If 5*x + 3 = 13, then x is the number of times the number of times the number of times the number of times
- unconditioned 0: <|bos|>If the high dollar dog market is going to go too far in ensuring that retailers maintain their sales, then what PEOPLE keep accounting for the rates to embrace inflation for years to come. Is foam marketing systems meant to prevent future demise for a specific figure? Will it work? What will the legacy pump need to stay like a wayward bureaucrat indefinitely? Techno profits and worries in 90 days is box I believe. Please let Arathenounce on marketing.
@David: Yeah it is much more out of Walmart with a 1/3 to 1/2 premium in the regular market. Same trouble may be
- unconditioned 1: <|bos|>Many glasses produced in the past decade, with new technology comes a futuristic, quality, and suitable model.

Choosing the Philippine lens for your glasses

The lens lens is an optical element, which functions as the mirror of a spectrometer. It indicates the proportions of wavelengths; it includes the optical optimal sharpness, refractive power, and OD, or the distance to the retina.

The design of the lens is depending on the structure and function of the deuterome, which is a thin flexible structure present in the whole of the lens. At the outer surface of the labyrinth, the ridges are eustaces of a curvature wave
- unconditioned 2: <|bos|>Your rotary isn't shaking something like a floppy disk look at how much it shakes. You're not actually shaking anything with the point. And no, a plastic pin causes that to happen. It's just a standard beach ball. Some women keep three of dolls at a time. And they don't show up after two OB/GYNs take the same long sample out of all to analyze the pieces exactly. The thought of that and the shaking isn't accelerated by the brakston air dryers (or the thing that cuts out the squeaking from the washing machine, you say) we had to look up and tell someone that it
- unconditioned 3: <|bos|>Changes in orthophysiology and conductive hearing loss in talented, spoken-hearing people cope well with losing contact with the
wound like a decision is in the best interest of people when going to music concerts.
For example at the scene of a final person's choice involving the violin various conversations actually occurs in discussions of the composer later than listening to the vocal parts. While there was no audio of the same musical background recent analytic views of sound changes were later in discussion in relation to noise changes or blood pellon in the right development of thought, with support from various sources (such as electronic sources such as W
- unconditioned 4: <|bos|>diapers company singaporean

In the vast ever-evolving business landscape, the need to stand out is pivotal. Inspiring young minds to soar into the skies of a successful career path sets the stage for business success. This is where eLip 4 CTUSZ® comes into play.

Our mission is to revolutionize the travel and tourism industry. We transcend trends, making our aviation pioneering efforts more sustainable and inclusive. Welcome to a future where luxury skywares meet the skies.

Today's rapidly evolving travel and tourism market demands a world where company meets the skies. 4 CTUSZ® Skywares: A Technological
- unconditioned 5: <|bos|>Using new outdoor education classes. The goal is to inspire students to connect with the natural world outdoors

Including home educating in remote learning sessions

Support in getting us fit

Networking with teams from a wide variety of forward

What will our elementary students be doing during the 1st grade period? They are loving science, reading, dedicating to their education with opportunities to make connections with the natural world around them, and exploring our local ecosystem. The goal is to enhance our educational program through our field trips, family involvement, digital classroom events, and student activites.

They will work in the garden on campus planting bulbs to get
- unconditioned 6: <|bos|>I want to explore conceptually for all 3rd grade students for lessons regarding space and matter. I have a 7th grade teacher who has used a pantom hose with his 2nd graders. We begin with Ola Muncie's Space Game. He's really good at the quantity and quality of time we do gather. He only needs one of the 8 objects from the pantom hose. At what point would he get his teacher to pause the boy band and purposely undersab a junkyard. This next lesson would talk about Primitive Spheres and why they were useful in Space Exploration. He gets
- unconditioned 7: <|bos|>Can Dogs Learn to Ride on a Cruise Car?

July 21, 2023

2 min read

Share

Link

Can Dogs Learn to Ride on a Cruise Car?

How do your dogs learn to ride on a cruise? A violent freight train will have niches above the cabin. What breed of dog rides on the cruise ship? Dogs of a different breed?

The fact that dogs were taught as puppies to ride on the rail and ship coasts means dogs are taught as children to ride all over, amusement parks, and public beaches for years! Our team provides their customers with all the info and tools they need to


## Chat evaluation sft
timestamp: 2026-06-08 22:37:55

- source: sft
- task_name: None
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.3956
- ARC-Challenge: 0.3251
- MMLU: 0.3178
- GSM8K: 0.0417
- HumanEval: 0.0915
- SpellingBee: 0.9922
- ChatCORE metric: 0.2517


## Summary

- Characters: 525,455
- Lines: 12,482
- Files: 51
- Tokens (approx): 131,363
- Dependencies (uv.lock lines): 4,897

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.1520   | -        | -        |
| ARC-Challenge   | -        | 0.3251   | -        |
| ARC-Easy        | -        | 0.3956   | -        |
| GSM8K           | -        | 0.0417   | -        |
| HumanEval       | -        | 0.0915   | -        |
| MMLU            | -        | 0.3178   | -        |
| ChatCORE        | -        | 0.2517   | -        |

Total wall clock time: 10h26m
