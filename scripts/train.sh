#!/bin/bash
deepspeed --hostfile config/hostfile \
--force_multi \
train.py \
--deepspeed \
--deepspeed_config config/deepspeed.json
