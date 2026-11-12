"""The machine-learning experiment of Deliverable 3 (needs PyTorch; the simulator itself does not).

    eca.py        facts about the 256 elementary rules: families, Wolfram classes, held-out rules
    data.py       loads a dataset made by `cellab dataset` and splits it
    models.py     the two networks: StepNet (one step) and AnswerNet (all five questions in one pass)
    train.py      python -m ml.train ...      trains either network, can resume
    evaluate.py   python -m ml.evaluate ...   metrics per question, class and horizon, with baselines
"""
