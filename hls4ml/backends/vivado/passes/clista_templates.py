from hls4ml.model.layers import CLISTA_Encoder
from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# CLISTA templates

conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

clista_config_template = """struct config{index} : nnet::clista_config {{
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const nnet::conv_implementation implementation = nnet::conv_implementation::{implementation};
    static const unsigned min_height = {min_height};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef {accum_t.name} accum_t;
    typedef {weight_t.name} weight_t;
    typedef {bias_t.name} bias_t;
    typedef {config_t} mult_config;

    static const unsigned n_iters = {n_iters};
    static const bool positive_code = strings_equal("{positive_code}", "True");
}};
const ap_uint<config{index}::filt_height * config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

clista_function_template = 'nnet::clista<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

clista_include_list = ['nnet_utils/nnet_clista.h', 'nnet_utils/nnet_clista_stream.h']

class CLISTAConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(CLISTA_Encoder)
        self.template = clista_config_template
        self.mult_template = conv_mult_config_template
    
    def format(self, node):
        params = self._default_config_params(node)
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('weight').nzeros

        params['config_t'] = 'config{}_mult'.format(node.index)
        conv_config = self.template.format(**params)

        mult_params = self._default_config_params(node)
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('weight').type.precision)
        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config

class CLISTAFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(CLISTA_Encoder, include_header=clista_include_list)
        self.template = clista_function_template
    
    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)