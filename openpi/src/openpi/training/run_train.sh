#!/bin/bash

tmux new-session -d -s openpi_train

tmux send-keys -t openpi_train " sh '/home/zwc/openpi/src/openpi/training/train_openpi.sh'" C-m

echo "  tmux attach-session -t openpi_train    # connect to tmux"
echo "  tmux detach                           # close tmux"
echo "  tmux kill-session -t openpi_train     # kill tmux"