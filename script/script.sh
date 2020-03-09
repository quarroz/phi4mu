#!/bin/bash

method='U1'  # Can be BDF,scipyRK or RK4

for mu in {0.1..0.9..0.1};
do
    for T in {0..5..0.5};
        do
            qsub /lpt/jquarroz/python/phi4mu/script/script0.sh $mu $T
        done
done

