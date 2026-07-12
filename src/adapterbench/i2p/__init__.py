"""Image-to-PEFT (i2p) setting: the reward-tilting image-domain counterpart to the language
`t2p`/`d2p` settings. A from-scratch adapter (LoRA baseline codec) is hooked into a frozen
SD-Turbo's attention and trained to modulate the initial noise toward higher reward
(HyperNoise; Eyring et al., 2025), reusing the same codec seam and the four benchmark
invariants. See PROJECT_PLAN.md's "Image domain" section."""
