for ((i=0;i<2;i+=1))
  do
  for mc in False;
  do
    for tree_samples  in 5;
    do
    for weight in 0.1;
      do
        for backprop in wass;
        do
            for selection in optimistic;
            do
              for difficulty in easy medium;
              do
                #echo std_weight_$weight
                python3 point.py --path_results /data/amarildo/wazero/results/ --tree_samples_ratio $tree_samples --suffix tree_samples_$tree_samples/std_weight_$weight/zero_init/std_lr_0.01 --std_lr 0.01 --sample_size 64 --test_size 20 --max_processes 10 --mem_max_capacity 500000 --n_epochs 400 --horizon 150 --depth 30 --difficulty $difficulty --n 7 --r_min -1 --r_max 0 --num_hidden 20 --mc_targets $mc --backpropagation $backprop --action_selection $selection --prv_std_qty 1. --prv_std_weight $weight --lr 0.001&
              done
            done
        done
      done
    done
  done
done