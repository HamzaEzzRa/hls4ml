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

#ifndef _NNET_CLISTA_STREAM_
#define _NNET_CLISTA_STREAM_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_conv2d.h"
// #include "nnet_conv2d_stream.h"
// #include "nnet_clista.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

// template<class data_T, class res_T>
// void accumulate_stream(hls::stream<data_T> &data, hls::stream<res_T> &res) {
//     AccumLoop: for (int i = 0; i < data_T::size / res_T::size; i++) {
//         #pragma HLS PIPELINE
        
//         data_T in_data = data.read();
//         res_T out_data = res.read();

//         PackLoop: for (int j = 0; j < res_T::size; j++) {
//             #pragma HLS UNROLL
//             out_data[j] = out_data[j] + in_data[j];
//         }

//         res.write(out_data);
//     }
// }

// template<class data_T, class res_T, typename CONFIG_T>
// void softshrink_stream(data_T data, res_T res)
// {
//     const int in_size = CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan;

//     ActLoop: for (int i = 0; i < in_size / res_T::size; i++) {
//         #pragma HLS PIPELINE
        
//         data_T in_data = data.read();
//         res_T out_data;
//         #pragma HLS DATA_PACK variable=out_data

//         PackLoop: for (int j = 0; j < res_T::size; j++) {
//             #pragma HLS UNROLL
//             if (in_data[j] > theta || in_data[j] < theta) out_data[j] = in_data[j];
//             else out_data[j] = 0;
//         }

//         res->write(out_data);
//     }
// };

// template<class data_T, class res_T, typename CONFIG_T>
// void zeroclip_stream(hls::stream<data_T> *data, hls::stream<res_T> &res)
// {
//     const int in_size = CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan;

//     ClipLoop: for (int i = 0; i < in_size / res_T::size; i++) {
//         #pragma HLS PIPELINE

//         data_T in_data = data->read();
//         res_T out_data;
//         #pragma HLS DATA_PACK variable=out_data

//         PackLoop: for (int j = 0; j < res_T::size; j++) {
//             #pragma HLS UNROLL
//             if (in_data[j] > 0) out_data[j] = in_data[j];
//             else out_data[j] = 0;
//         }

//         res.write(out_data);
//     }
// };

template<class data_T, class res_T, typename CONFIG_T>
void clista_latency(
    hls::stream<data_T> &data_stream,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    const int in_size = CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan;
    const int out_size = CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt;
    
    typename data_T::value_type data[in_size];
    typename res_T::value_type res[out_size];
    
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=res complete

    DataPrepare: for (int i_in = 0; i_in < in_size / data_T::size; i_in++) {
        if (in_size / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    typename data_T::value_type tmp[in_size];
    typename data_T::value_type acc[out_size];

    // #pragma HLS ARRAY_PARTITION variable=tmp complete
    // #pragma HLS ARRAY_PARTITION variable=acc complete

    typename data_T::value_type cache;
    Activation1: for (int iacc = 0; iacc < in_size; iacc++) {
        cache = data[iacc];
        tmp[iacc] = CONFIG_T::template softshrink<typename data_T::value_type, typename data_T::value_type>::activation(cache, CONFIG_T::theta);
    }

    if (CONFIG_T::n_iters > 1) {
        Product1: for (int kk = 0; kk < CONFIG_T::n_iters; kk++) {
            for (int iacc = 0; iacc < in_size; iacc++) {
                tmp[iacc] = tmp[iacc] + data[iacc];
            }

            nnet::conv_2d_cl<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(tmp, acc, weights, biases);

            Activation2: for (int iacc = 0; iacc < out_size; iacc++) {
                tmp[iacc] = CONFIG_T::template softshrink<typename data_T::value_type, typename data_T::value_type>::activation(acc[iacc], CONFIG_T::theta);
            }
        }
    }

    if (CONFIG_T::positive_code) {
        Clip1: for (int ii = 0; ii < out_size; ii++) {
            res[ii] = CONFIG_T::template zeroclip<typename data_T::value_type>::activation(tmp[ii]);
        }
    }

    ResWrite: for(unsigned i_out = 0; i_out < out_size / res_T::size; i_out++) {
        if (out_size / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        #pragma HLS DATA_PACK variable=res_pack
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }

    // const int in_size = CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan;
    // const int out_size = CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt;
    
    // hls::stream<data_T> *tmp = new hls::stream<data_T>();
    // hls::stream<res_T> acc;

    // #pragma HLS ARRAY_PARTITION variable=tmp complete
    // #pragma HLS ARRAY_PARTITION variable=acc complete

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    // #pragma HLS function_instantiate variable=weights,biases

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    // #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    // #pragma HLS ARRAY_PARTITION variable=biases complete
    // #pragma HLS ARRAY_PARTITION variable=tmp complete
    // #pragma HLS ARRAY_PARTITION variable=acc complete

    // softshrink_stream<data_T, data_T, CONFIG_T>(data, tmp, CONFIG_T::theta);

    // if (CONFIG_T::n_iters > 1) {
        // Product1: for (int kk = 0; kk < CONFIG_T::n_iters; kk++) {
            // #pragma HLS UNROLL
            // nnet::conv_2d_cl<data_T, res_T, CONFIG_T>(tmp, acc, weights, biases);

            // accumulate_stream<data_T, res_T>(data, acc);

            // softshrink_stream<res_T, data_T, typename CONFIG_T::weight_t>(acc, tmp, CONFIG_T::theta);
        // }
    // }

    // if (CONFIG_T::positive_code) {
    //     zeroclip_stream()
    // }

    // Cast to "res_t" type
    // Result: for (int ires = 0; ires < CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt; ires++) {
    //     res[ires] = cast<data_T, res_T, CONFIG_T>(tmp[ires]);
    // }
}

template<class data_T, class res_T, typename CONFIG_T>
void clista_resource(
    hls::stream<data_T> &data_stream,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    const int in_size = CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan;
    const int out_size = CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt;
    
    typename data_T::value_type data[in_size];
    typename res_T::value_type res[out_size];
    
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=res complete

    DataPrepare: for (int i_in = 0; i_in < in_size / data_T::size; i_in++) {
        if (in_size / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    typename data_T::value_type tmp[in_size];
    typename data_T::value_type acc[out_size];

    // #pragma HLS ARRAY_PARTITION variable=tmp complete
    // #pragma HLS ARRAY_PARTITION variable=acc complete

    typename data_T::value_type cache;
    Activation1: for (int iacc = 0; iacc < in_size; iacc++) {
        cache = data[iacc];
        tmp[iacc] = CONFIG_T::template softshrink<typename data_T::value_type, typename data_T::value_type>::activation(cache, CONFIG_T::theta);
    }

    if (CONFIG_T::n_iters > 1) {
        Product1: for (int kk = 0; kk < CONFIG_T::n_iters; kk++) {
            for (int iacc = 0; iacc < in_size; iacc++) {
                tmp[iacc] = tmp[iacc] + data[iacc];
            }

            nnet::conv_2d_cl<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(tmp, acc, weights, biases);

            Activation2: for (int iacc = 0; iacc < out_size; iacc++) {
                tmp[iacc] = CONFIG_T::template softshrink<typename data_T::value_type, typename data_T::value_type>::activation(acc[iacc], CONFIG_T::theta);
            }
        }
    }

    if (CONFIG_T::positive_code) {
        Clip1: for (int ii = 0; ii < out_size; ii++) {
            res[ii] = CONFIG_T::template zeroclip<typename data_T::value_type>::activation(tmp[ii]);
        }
    }

    ResWrite: for(unsigned i_out = 0; i_out < out_size / res_T::size; i_out++) {
        if (out_size / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        #pragma HLS DATA_PACK variable=res_pack
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void clista(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
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

#endif // _NNET_CLISTA_STREAM_