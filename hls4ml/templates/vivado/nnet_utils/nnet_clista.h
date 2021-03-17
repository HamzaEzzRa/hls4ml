//    Hamza Ezzaoui Rahali, 2022-2023
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

#ifndef _NNET_CLISTA_
#define _NNET_CLISTA_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_conv2d.h"
#include "nnet_lista.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void clista_latency(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    data_T cache;
    typename CONFIG_T::accum_t tmp[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=tmp complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=tmp complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    Activation1: for (int iacc = 0; iacc < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan; iacc++) {
        cache = data[iacc];
        tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(cache, CONFIG_T::theta);
    }

    if (CONFIG_T::n_iters > 1) {
        Product1: for (int kk = 0; kk < CONFIG_T::n_iters; kk++) {
            nnet::conv_2d_cl<data_T, res_T, CONFIG_T>(tmp, acc, weights, biases);

            Activation2: for (int iacc = 0; iacc < CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt; iacc++) {
                tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(data[iacc] + acc[iacc], CONFIG_T::theta);
            }
        }
    }

    if (CONFIG_T::positive_code) {
        Clip1: for (int ii = 0; ii < CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt; ii++) {
            tmp[ii] = CONFIG_T::template zeroclip<data_T>::activation(tmp[ii]);
        }
    }

    // Cast to "res_t" type
    Result: for (int ires = 0; ires < CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(tmp[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void clista_resource(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin*nout, rufactor);

    const int in_size = CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan;
    const int out_size = CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt;

    // #pragma HLS function_instantiate variable=weights,biases
    // Commenting out the deisgnation HLS seems to choose correctly
    // #pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM
    // #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    // #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t tmp[in_size];
    typename CONFIG_T::accum_t acc[out_size];
    #pragma HLS ARRAY_PARTITION variable=tmp complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    Activation1:
    for (int iacc = 0; iacc < in_size; iacc++) {
        #pragma HLS UNROLL
        tmp[iacc] = static_cast<typename CONFIG_T::accum_t>(
            CONFIG_T::template softshrink<data_T, data_T>::activation(
                data[iacc], CONFIG_T::theta));
    }

    ReuseLoop:
    if (CONFIG_T::n_iters > 1) {
        for (int iter=0; iter < CONFIG_T::n_iters; iter++) {
            nnet::conv_2d_cl<data_T, res_T, CONFIG_T>(tmp, acc, weights, biases);

            Activation2: for (int iacc = 0; iacc < out_size; iacc++) {
                #pragma HLS UNROLL

                // acc[iacc] += (typename CONFIG_T::accum_t) biases[iacc];
                tmp[iacc] = CONFIG_T::template softshrink<data_T, data_T>::activation(data[iacc] + acc[iacc], CONFIG_T::theta);
                // acc[iacc] = 0;
            }
        }
    }

    if (CONFIG_T::positive_code) {
        Clip1: for (int ii = 0; ii < out_size; ii++) {
            #pragma HLS UNROLL
            tmp[ii] = CONFIG_T::template zeroclip<data_T>::activation(tmp[ii]);
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < out_size; ires++) {
        // #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(data[ires]);
    }
}

struct clista_config
{
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Convolutional parameters
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 1;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 1;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 10;
    static const unsigned out_width = 10;
    static const unsigned dilation_height = 1;
    static const unsigned dilation_width = 1;

    // Resource reuse info
    // static const unsigned io_type = io_parallel;
    // static const unsigned strategy = latency; 
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;

    // Additional LISTA parameters
    static constexpr weight_t theta = 0.0;
    static const unsigned n_iters = 5;
    static const bool positive_code = false;

    // Additional used functions
    template<class x_T, class y_T>
    using softshrink = nnet::softshrink<x_T, y_T>;

    template<class x_T>
    using zeroclip = nnet::zeroclip<x_T>;
};

template<class data_T, class res_T, typename CONFIG_T>
void clista(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    if (CONFIG_T::strategy == nnet::latency) {
        clista_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        clista_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

}

#endif // _NNET_CLISTA_