#!/bin/bash

tmux new-session -d -s openpi_train2

tmux send-keys -t openpi_train2 " sh '/home/zwc/openpi/src/openpi/training/train_openpi2.sh'" C-m

echo "  tmux attach-session -t openpi_train2    # connect to tmux"
echo "  tmux detach                           # close tmux"
echo "  tmux kill-session -t openpi_train2     # kill tmux"