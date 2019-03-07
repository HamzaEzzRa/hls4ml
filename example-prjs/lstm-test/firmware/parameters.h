#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_sublayer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_recursive.h"
#include "nnet_activation.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<32,8> accum_default_t;
typedef ap_fixed<32,8> weight_default_t;
typedef ap_fixed<32,8> bias_default_t;
typedef ap_fixed<32,8> input_t;
typedef ap_fixed<32,8> result_t;
#define N_LOOP    20
#define N_INPUTS  6
#define N_LAYER_1 16
#define N_STATE_1 16
#define N_OUTPUTS 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<32,8> layer1_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::lstm_config {
        typedef accum_default_t accum_t;
        typedef weight_default_t weight_t;  // Matrix
        typedef bias_default_t   bias_t;  // Vector
        static const unsigned n_in  = N_INPUTS;
        static const unsigned n_out = N_LAYER_1;
        static const unsigned n_state = N_STATE_1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned activation_type = nnet::activ_relu;
        static const unsigned reuse_factor = 1;
        static const bool     store_weights_in_bram = false;        
        };
struct tanh_config1 : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1*4;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct tanh_config1_lstm : nnet::activ_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config2 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_1;
        static const unsigned n_out = N_OUTPUTS;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 2;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct softmax_config2 : nnet::activ_config {
        static const unsigned n_in = N_OUTPUTS;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };

#endif 