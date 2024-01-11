for ((i=0;i<2;i+=1))
do
  for difficulty in easier easy medium;
  do
      #echo "$bits $selection $backprop $mc"
      python3 point.py --path_results /data/amarildo/wazero/results_easier/  --sample_size 64 --test_size 20 --max_processes 10 --mem_max_capacity 500000 --n_epochs 200 --depth 50 --horizon 80 --distance_reward False --difficulty $difficulty --n 7 --num_hidden 20  --optimistic False --lr 0.001&
  done
done