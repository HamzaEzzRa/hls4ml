from hls4ml.model.layers import LISTA_Block
from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# LISTA templates

# for parametric booleans (ex: positive_code), Python will pass
# True/False (uppercase) which is invalid syntax in C++ hence the ternary op
lista_config_template = """struct config{index} : nnet::lista_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;
    static const unsigned n_iters = {n_iters};
    static const bool positive_code = !strcmp("{positive_code}", "True") ? true : false;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {index_t.name} index_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""
    # template<class x_T, class y_T>
    # using softshrink = nnet::{softshrink_type}<x_T, y_T>;
    # template<class x_T>
    # using zeroclip = nnet::{zeroclip_type}<x_T>;
#}};\n"""

lista_function_template = 'nnet::lista<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

lista_include_list = ['nnet_utils/nnet_lista.h']

class LISTAConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(LISTA_Block)
        self.template = lista_config_template
    
    def format(self, node):
        params = self._default_config_params(node)
        params['nzeros'] = node.get_weights('weight').nzeros
        params['nonzeros'] = node.get_weights('weight').nonzeros
        params['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('weight').type.precision)
        # params['softshrink_type'] = get_backend('vivado').softshrink_type(...)
        # params['zeroclip_type'] = get_backend('vivado').zeroclip_type(...)

        return self.template.format(**params)

class LISTAFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(LISTA_Block, include_header=lista_include_list)
        self.template = lista_function_template
    
    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        # params['theta'] = node.get_weights('theta').name

        return self.template.format(**params)