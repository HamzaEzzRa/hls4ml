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
            #pragma HLS PIPELINE
        }
        cache = data[iacc];
        tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(cache, CONFIG_T::theta);
    }

    if (CONFIG_T::n_iters > 1) {
        Product1: for (int kk = 0; kk < CONFIG_T::n_iters; kk++) {
            // if (CONFIG_T::io_type == io_serial) {
            //     #pragma HLS PIPELINE
            // }
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
                    #pragma HLS PIPELINE
                }
                acc[iacc] += (typename CONFIG_T::accum_t) biases[iacc];
                tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(data[iacc] + acc[iacc], CONFIG_T::theta);
                acc[iacc] = 0;
            }
        }
    }

    if (CONFIG_T::positive_code) {
        Clip1: for (int ii = 0; ii < CONFIG_T::n_out; ii++) {
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
void lista_resource(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    // ...
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