"""The council's model, built from scratch: no outside AI, no pretrained weights.

    autograd.py      reverse-mode autodiff on NumPy
    tokenizer.py     byte-level BPE, trained on the corpus
    transformer.py   GPT-style decoder (the "hidden layers")
    trainer.py       corpus, AdamW, checkpoints, the training loop
    brain.py         the trained model at inference time
    hardware_detect  picks a model size for this machine
"""
