for ((i=0;i<2;i+=1))
  do
  for mc in False;
  do
    for tree_samples  in 0;
    do
    for weight in 0.;
      do
        for backprop in wass optimistic;
        do
            for selection in optimistic counts;
            do
              for difficulty in easier easy medium;
              do
                #echo std_weight_$weight
                python3 point.py --path_results /data/amarildo/wazero/results_easier/ --tree_samples_ratio $tree_samples --suffix tree_samples_$tree_samples/std_weight_$weight/zero_init/std_lr_0.01 --std_lr 0.01 --sample_size 64 --test_size 20 --max_processes 10 --mem_max_capacity 500000 --n_epochs 200 --horizon 80 --depth 50 --difficulty $difficulty --n 7 --r_min -0.1 --r_max 0 --num_hidden 20 --mc_targets $mc --backpropagation $backprop --action_selection $selection --prv_std_qty 1. --prv_std_weight $weight --lr 0.001 --distance_reward False&
              done
            done
        done
      done
    done
  done
done