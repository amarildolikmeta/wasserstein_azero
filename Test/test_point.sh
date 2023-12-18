for ((i=0;i<3;i+=1))
  do
  for mc in False;
  do
    for tree_samples  in 0 5 10;
    do
    for weight in 0. 0.1;
      do
        for backprop in wass optimistic;
        do
            for selection in counts optimistic;
            do
              for difficulty in easy medium hard;
              do
                #echo std_weight_$weight
                python3 point.py --path_results /data/amarildo/wazero/results/ --tree_samples_ratio $tree_samples --suffix tree_samples_$tree_samples/std_weight_$weight --sample_size 64 --test_size 20 --max_processes 10 --mem_max_capacity 500000 --n_epochs 300 --horizon 200 --depth 70 --difficulty $difficulty --n 7 --r_min -0.1 --r_max 0 --num_hidden 64 --mc_targets $mc --backpropagation $backprop --action_selection $selection --prv_std_qty 1. --prv_std_weight $weight --lr 0.001
              done
            done
        done
      done
    done
  done &
done