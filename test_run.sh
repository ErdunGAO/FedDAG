
echo "Started running experiments."

python test.py  --seed 2022 \
                --node 10 \
                --edge 20 \
                --num_client 10 \
                --n 100 \
                --graph_type er \
                --gen_method multiiid \
                --linearity linear \
                --fed_type AS_linear 

python test.py  --seed 2022 \
                --node 10 \
                --edge 20 \
                --num_client 10 \
                --n 600 \
                --graph_type er \
                --gen_method multiiid \
                --linearity nonlinear \
                --fed_type AS \
                --sem_type gp \
                --init_rho 6e-3 \
                --rho_multiply 10 \
                --l1_graph_penalty 0.01 \
                --use_gpu

python test.py  --seed 2022 \
                --node 10 \
                --edge 20 \
                --num_client 10 \
                --n 600 \
                --graph_type er \
                --gen_method noniid \
                --linearity nonlinear \
                --fed_type GS \
                --init_rho 6e-3 \
                --rho_multiply 10 \
                --l1_graph_penalty 0.01 \
                --use_gpu