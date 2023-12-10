for ((i=0;i<3;i+=1))
  do
  for mc in False True;
  do
    for weight in 0.1 1.;
    do
    for backprop in optimistic mc;
      do
        for selection in counts optimistic;
        do
            for bits in 10 12 15 18;
            do
              #echo std_weight_$weight
              python3 bitflip.py --path_results /data/amarildo/wazero/results/ --suffix std_weight_$weight --sample_size 50 --test_size 20 --max_processes 10 --mem_max_capacity 200000 --n_epochs 200 --depth 32 --bit_depth $bits --n 7 --r_min -0.1 --r_max 0 --num_hidden 20 --mc_targets $mc --backpropagation $backprop --action_selection $selection --prv_std_qty 1. --prv_std_weight $weight --lr 0.0001
            done
        done
      done
    done
  done &
done