for ((i=0;i<3;i+=1))
do
  for bits in 18;
  do
      #echo "$bits $selection $backprop $mc"
      python3 bitflip.py --path_results /data/amarildo/wazero/results/  --sample_size 50 --test_size 20 --max_processes 10 --mem_max_capacity 200000 --n_epochs 300 --depth 32 --bit_depth $bits --n 7 --num_hidden 20  --optimistic False --lr 0.001
  done &
done