#!/bin/bash

source ~/.virtualenvs/llm/bin/activate

P=workspace/plots

export PROB_EVAL=false

python $P/plotly_leaderboard.py
python $P/plotly_bydim.py
python $P/plotly_bydomain.py
python $P/plotly_bydata.py
python $P/plotly_bysize.py
python www/gen_ia_table.py
python www/db_create.py

export PROB_EVAL=true

python $P/plotly_leaderboard.py
python $P/plotly_bydim.py
python $P/plotly_bydomain.py
python $P/plotly_bydata.py
python $P/plotly_bysize.py

deactivate

