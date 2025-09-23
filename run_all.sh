#!/bin/bash

source ~/.virtualenvs/llm/bin/activate

for f in workspace/plots/*.py; do
  python "$f"
done

for f in www/*.py; do
  python "$f"
done

deactivate

