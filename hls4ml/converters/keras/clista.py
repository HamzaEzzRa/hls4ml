from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.utils import parse_data_format, compute_padding_2d
from hls4ml.converters.keras.qkeras import get_quantizer_from_config

@keras_handler('CLISTA_Encoder')
def parse_clista_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('CLISTA_Encoder' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    (
        layer['in_height'],
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], layer['data_format'])

    if 'filters' in keras_layer['config']:
        layer['n_filt'] = keras_layer['config']['filters']
    else:    
        layer['n_filt'] = layer['n_chan']
    layer['filt_height'] = keras_layer['config']['kernel_size'][0]
    layer['filt_width'] = keras_layer['config']['kernel_size'][1]
    layer['stride_height'] = keras_layer['config']['strides'][0]
    layer['stride_width'] = keras_layer['config']['strides'][1]
    layer['padding'] = keras_layer['config']['padding']
    
    (
        layer['out_height'],
        layer['out_width'],
        layer['pad_top'],
        layer['pad_bottom'],
        layer['pad_left'],
        layer['pad_right']
    ) = compute_padding_2d(
        layer['padding'],
        layer['in_height'],
        layer['in_width'],
        layer['stride_height'],
        layer['stride_width'],
        layer['filt_height'],
        layer['filt_width']
    )

    layer['L'] = keras_layer['config']['L']
    layer['theta'] = keras_layer['config']['theta']
    layer['n_iters'] = keras_layer['config']['n_iters']
    layer['positive_code'] = keras_layer['config']['positive_code']

    if layer['data_format'] == 'channels_first':
        output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
    else:
        output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]

    return layer, output_shape

@keras_handler('QCLISTA_Encoder')
def parse_qclista_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('QCLISTA_Encoder' in keras_layer['class_name'])

    layer, output_shape = parse_clista_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    if keras_layer['config']['bias_quantizer'] is not None:
        layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')
    else:
        layer['bias_quantizer'] = None
    
    return layer, output_shape