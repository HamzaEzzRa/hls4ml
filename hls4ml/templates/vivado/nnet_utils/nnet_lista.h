//    Copyright (C) 2022 Hamza Ezzaoui Rahali
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program. If not, see <http://www.gnu.org/licenses/>.
//

#ifndef _NNET_LISTA_
#define _NNET_LISTA_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

constexpr bool strings_equal(char const * a, char const * b) {
    return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

namespace nnet {

template<class x_T, class theta_T>
class softshrink
{
    public:
    static auto activation(x_T a, theta_T theta) -> decltype(a)
    {
        // #pragma HLS INLINE
        return a > theta || a < -theta ? a : (x_T)0;
    }
};

template<class x_T>
class zeroclip
{
    public:
    static auto activation(x_T a) -> decltype(a)
    {
        #pragma HLS INLINE
        return a > (x_T)0 ? a : (x_T)0;
    }
};

template<class data_T, class res_T, typename CONFIG_T>
void lista_latency(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t tmp[CONFIG_T::n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel || CONFIG_T::io_type == io_stream) {
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        // #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=tmp complete
        #pragma HLS ARRAY_PARTITION variable=acc complete

        int multiplier_limit = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
        CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::limit(multiplier_limit);
    } else if (CONFIG_T::io_type == io_serial) {
        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
        // Otherwise, HLS wont be happy
        int cycle_factor = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        int reused_cycle = DIV_ROUNDUP(CONFIG_T::n_out, CONFIG_T::reuse_factor);
        if (cycle_factor != reused_cycle) {
            cycle_factor = CONFIG_T::n_out;
        }
        /*int cycle_factor = CONFIG_T::n_out;
        float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        if (reused_cycle == ceil(reused_cycle)){
            // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
            cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
        }*/
        #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=tmp complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=tmp depth=1
        #pragma HLS STREAM variable=acc depth=1
        if (CONFIG_T::store_weights_in_bram){
            #pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
        }
    }

    Activation1: for (int iacc = 0; iacc < CONFIG_T::n_in; iacc++) {
        if (CONFIG_T::io_type == io_serial) {
            #pragma HLS UNROLL
        }
        cache = data[iacc];
        tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(cache, CONFIG_T::theta);
    }

    if (CONFIG_T::n_iters > 1) {
        Product1: for (int kk = 0; kk < CONFIG_T::n_iters; kk++) {
            if (CONFIG_T::io_type == io_serial) {
                #pragma HLS PIPELINE
            }
            Product2: for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
                if (CONFIG_T::io_type == io_serial) {
                    #pragma HLS PIPELINE
                }
                cache = tmp[ii];
                Product3: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
                    if (CONFIG_T::io_type == io_serial) {
                        int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                        CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::limit(multiplier_limit);
                    }
                    int index = ii*CONFIG_T::n_out+jj;
                    acc[jj] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(cache, weights[index]);
                }
            }

            Activation2: for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
                if (CONFIG_T::io_type == io_serial) {
                    #pragma HLS UNROLL
                }
                acc[iacc] += (typename CONFIG_T::accum_t) biases[iacc];
                tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(data[iacc] + acc[iacc], CONFIG_T::theta);
                acc[iacc] = 0;
            }
        }
    }

    if (CONFIG_T::positive_code) {
        Clip1: for (int ii = 0; ii < CONFIG_T::n_out; ii++) {
            if (CONFIG_T::io_type == io_serial){
                #pragma HLS UNROLL
            }
            tmp[ii] = CONFIG_T::template zeroclip<data_T>::activation(tmp[ii]);
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = cast<data_T, res_T, CONFIG_T>(tmp[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void lista_resource_rf_leq_nin(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out]
)
{
    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

    #pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t tmp[CONFIG_T::n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    Activation1:
    for (int iacc = 0; iacc < CONFIG_T::n_in; iacc++) {
        #pragma HLS UNROLL

        tmp[iacc] = static_cast<typename CONFIG_T::accum_t>(
            CONFIG_T::template softshrink<data_T, data_T>::activation(
                data[iacc], CONFIG_T::theta));
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;

        MultLoop:
        if (CONFIG_T::n_iters > 1) {
            for (int iter=0; iter < CONFIG_T::n_iters; iter++) {
                for (int im = 0; im < block_factor; im++) {
                    #pragma HLS UNROLL

                    // Dot product
                    acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                    CONFIG_T::template product<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t>::product(
                        tmp[in_index], weights[w_index]));

                    // Increment w_index
                    w_index += rufactor;
                    // Increment in_index
                    in_index += rufactor;
                    if (in_index >= nin) {
                        in_index = ir;
                    }
                    // Increment out_index
                    if (acc_step + 1 >= multscale) {
                        acc_step = 0;
                        out_index++;
                    } else {
                        acc_step++;
                    }
                }

                Activation2: for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
                    #pragma HLS UNROLL

                    acc[iacc] += (typename CONFIG_T::accum_t) biases[iacc];
                    tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(data[iacc] + acc[iacc], CONFIG_T::theta);
                    acc[iacc] = 0;
                }
            }
        }
    }

    if (CONFIG_T::positive_code) {
        Clip1: for (int ii = 0; ii < CONFIG_T::n_out; ii++) {
            #pragma HLS UNROLL
            tmp[ii] = CONFIG_T::template zeroclip<data_T>::activation(tmp[ii]);
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(tmp[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void lista_resource_rf_gt_nin_rem0(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out]
)
{

}

template<class data_T, class res_T, typename CONFIG_T>
void lista_resource_rf_gt_nin(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out]
)
{

}

template<class data_T, class res_T, typename CONFIG_T>
void lista_resource(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    #pragma HLS INLINE region

    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        lista_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        lista_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        lista_resource_rf_gt_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

struct lista_config
{
    // Internal data type definitions
    typedef float weight_t;
    typedef float accum_t;

    // Layer sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency; 
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;

    // Additional LISTA parameters
    static constexpr weight_t theta = 0.0;
    static const unsigned n_iters = 5;
    static const bool positive_code = false;

    // Product function to use
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;

    // Additional functions to use
    template<class x_T, class y_T>
    using softshrink = nnet::softshrink<x_T, y_T>;

    template<class x_T>
    using zeroclip = nnet::zeroclip<x_T>;
};

template<class data_T, class res_T, typename CONFIG_T>
void lista(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    #pragma HLS inline
    if (CONFIG_T::strategy == nnet::latency) {
        lista_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        lista_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

}

#endif // _NNET_LISTA_