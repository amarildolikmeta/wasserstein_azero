for ((i=0;i<3;i+=1))
do
  for bits in 10 12 15 18;
  do
      #echo "$bits $selection $backprop $mc"
      python3 bitflip.py --sample_size 50 --test_size 20 --max_processes 10 --mem_max_capacity 200000 --n_epochs 200 --depth 32 --bit_depth $bits --n 7 --num_hidden 20  --optimistic False --lr 0.0001
  done &
done