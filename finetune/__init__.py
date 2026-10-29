"""Fine-tuning a small local model on verified CadQuery builds (#46).

    python -m finetune.make_requests --target 600     # part requests, written by a large model
    python -m finetune.collect --budget 300           # teacher builds; keeps the valid, right-size ones
    python -m finetune.export                         # train/val chat files for training

Training itself runs in the separate `finetune` pixi environment (PyTorch, CUDA), never in the app's.
"""
