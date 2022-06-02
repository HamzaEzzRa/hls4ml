import numpy as np

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

from hls4ml.model.types import Quantizer
from hls4ml.model.types import IntegerPrecisionType

class BinaryQuantizer(Quantizer):
    def __init__(self, bits=2):
        if bits == 1:
            hls_type = IntegerPrecisionType(width=1, signed=False)
        elif bits == 2:
            hls_type = IntegerPrecisionType(width=2)
        else:
            raise Exception('BinaryQuantizer suppots 1 or 2 bits, but called with bits={}'.format(bits))
        super(BinaryQuantizer, self).__init__(bits, hls_type)
    
    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        quant_data = data
        if self.bits == 1:
            quant_data = np.where(data > 0, ones, zeros).astype('int')
        if self.bits == 2:
            quant_data = np.where(data > 0, ones, -ones)
        return quant_data

class TernaryQuantizer(Quantizer):
    def __init__(self):
        super(TernaryQuantizer, self).__init__(2, IntegerPrecisionType(width=2))
    
    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        return np.where(data > 0.5, ones, np.where(data <= -0.5, -ones, zeros))

@keras_handler('LISTA_Block')
def parse_lista_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('LISTA_Block' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')

    layer['n_in'] = weights_shape[0]
    layer['n_out'] = weights_shape[1]

    if 'Binary' in layer['class_name']:
        layer['weight_quantizer'] = BinaryQuantizer(bits=2)
        layer['bias_quantizer'] = BinaryQuantizer(bits=2)
    elif 'Ternary' in layer['class_name']:
        layer['weight_quantizer'] = TernaryQuantizer()
        layer['bias_quantizer'] = TernaryQuantizer()
    else:
        layer['weight_quantizer'] = None
        layer['bias_quantizer'] = None

    layer['L'] = keras_layer['config']['L']
    layer['theta'] = keras_layer['config']['theta']
    layer['n_iters'] = keras_layer['config']['n_iters']
    layer['positive_code'] = keras_layer['config']['positive_code']

    output_shape = input_shapes[0][:]
    output_shape[-1] = layer['n_out']

    return layer, output_shape