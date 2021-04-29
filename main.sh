#!/bin/bash
#7.2.21
#bash run_local.sh beta100 main 100 i_max=100

# 22.2.21 # imax = number workers
#bash run_local.sh beta_lin main 200 i_max=200
#bash run_local.sh beta_lin_dim main 400 i_max=400

# 3.3.21
#bash run_local.sh beta_lin_dim2 main 100 i_max=100

# 7.3.21
# bash run_local.sh beta_lin_dim3 main 630 i_max=630

# 14.3.
#bash run_local.sh low_high_tau_x_rho main 180 i_max=180
#bash run_local.sh low_high_tau_x_rho2 main 180 i_max=180
#bash run_local.sh low_high_tau_x_rho3 main 180 i_max=180
#bash run_local.sh low_high_tau_x_rho3_long main 180 i_max=180
# how does tau_ou affect MSE -> not ran through
#bash run_local.sh low_high_tau_x_rho3_longlong main 720 i_max=720
# how does gamma affect MSE -> not stable
#bash run_local.sh low_high_tau_x_rho3_long_gamma main 180 i_max=180

# 20.3.
# how does beta=0.1 and gamma do? (with gamma != g0)
# bash run_local.sh low_high_tau_x_rho3_long_beta_smaller main 180 i_max=180
# how does beta=0.1 compare to beta=0.3, with gamma == g0
#bash run_local.sh low_high_tau_x_rho3_long_beta_smaller_g0 main 180 i_max=180
# how does a longer (tau_ou = 300s) time do, gamma != g0?
#bash run_local.sh low_high_tau_x_rho3_longlong2 main 720 i_max=720

#10.4.
# how does normed kernels affect the sweep?
#bash run_local.sh int_tau_const main 180 i_max=180
#bash run_local.sh int_tau_const_mini main 180 i_max=180

#11.4.
# repaired the kernel
bash run_local.sh int_tau_const_mini2 main 360 i_max=360
