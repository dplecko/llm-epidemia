#!/bin/bash

source ~/.virtualenvs/llm/bin/activate

P=workspace/plots

python $P/plt_high_dim.py
python $P/plt_lead.py
python $P/plt_bydata.py
python $P/plt_base_instruct.py
python $P/plt_closed.py
python $P/plt_likelihood.py

deactivate

