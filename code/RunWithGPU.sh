#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python deep_mlp.py
