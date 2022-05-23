import pytest
import numpy as np
import qkeras
import hls4ml

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Ones
from qkeras.quantizers import quantized_bits, quantized_relu, ternary, binary

from pathlib import Path

test_root_path = Path(__file__).parent

@pytest.mark.parametrize(
    'quantizer', [
        ('binary'),
        ('ternary'),
        ('quantized_bits(8, 2)'),
        ('quantized_relu(8, 2)')
    ]
)
def test_qactivation_kwarg(quantizer):
    inputs = Input(shape=(10,))
    outputs = qkeras.QDense(
        10,
        kernel_quantizer=qkeras.quantized_bits(8, 2),
        activation=quantizer
    )(inputs)

    model = Model(inputs, outputs)
    out_dir = str(test_root_path / 'hls4mlprj_qactivation_kwarg_{}'.format(
        quantizer)
    )
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=out_dir
    )
    hls_model.compile()
