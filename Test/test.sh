for ((i=0;i<1;i+=1))
  do
  for mc in False;
  do
    for weight in 0.;
    do
    for backprop in wass optimistic;
      do
        for selection in counts;
        do
            for tree_samples in 0 10;
            do
              for std_lr in 0.005 0.01;
              do
                for gamma in 0.99;
                do
                  for bits in 10;
                  do
                    #echo std_weight_$weight
                    python3 bitflip.py --path_results /data/amarildo/wazero/results_2/  --gamma $gamma --tree_samples_ratio $tree_samples --suffix tree_samples_$tree_samples/std_weight_$weight --sample_size 50 --test_size 20 --max_processes 10 --mem_max_capacity 200000 --n_epochs 300 --depth 32 --bit_depth $bits --n 7 --r_min -0.1 --r_max 0 --num_hidden 20 --mc_targets $mc --backpropagation $backprop --action_selection $selection --prv_std_qty 1. --prv_std_weight $weight --lr 0.001 --std_lr $std_lr &
                  done
                done
              done
            done
        done
      done
    done
  done
done