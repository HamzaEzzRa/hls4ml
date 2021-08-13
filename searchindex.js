Search.setIndex({docnames:["api/configuration","api/hls-model","api/profiling","autodoc/hls4ml","autodoc/hls4ml.converters","autodoc/hls4ml.converters.keras","autodoc/hls4ml.converters.onnx","autodoc/hls4ml.converters.pytorch","autodoc/hls4ml.model","autodoc/hls4ml.model.optimizer","autodoc/hls4ml.model.optimizer.passes","autodoc/hls4ml.report","autodoc/hls4ml.templates","autodoc/hls4ml.utils","autodoc/hls4ml.writer","command","concepts","index","reference","release_notes","setup","status"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,sphinx:56},filenames:["api/configuration.rst","api/hls-model.rst","api/profiling.rst","autodoc/hls4ml.rst","autodoc/hls4ml.converters.rst","autodoc/hls4ml.converters.keras.rst","autodoc/hls4ml.converters.onnx.rst","autodoc/hls4ml.converters.pytorch.rst","autodoc/hls4ml.model.rst","autodoc/hls4ml.model.optimizer.rst","autodoc/hls4ml.model.optimizer.passes.rst","autodoc/hls4ml.report.rst","autodoc/hls4ml.templates.rst","autodoc/hls4ml.utils.rst","autodoc/hls4ml.writer.rst","command.rst","concepts.rst","index.rst","reference.rst","release_notes.rst","setup.rst","status.rst"],objects:{"":{hls4ml:[3,0,0,"-"]},"hls4ml.converters":{convert_from_config:[4,1,1,""],convert_from_keras_model:[4,1,1,""],convert_from_onnx_model:[4,1,1,""],convert_from_pytorch_model:[4,1,1,""],keras:[5,0,0,"-"],keras_to_hls:[4,0,0,"-"],onnx:[6,0,0,"-"],onnx_to_hls:[4,0,0,"-"],parse_yaml_config:[4,1,1,""],pytorch:[7,0,0,"-"],utils:[4,0,0,"-"]},"hls4ml.converters.keras":{convolution:[5,0,0,"-"],core:[5,0,0,"-"],graph:[5,0,0,"-"],merge:[5,0,0,"-"],pooling:[5,0,0,"-"],reshape:[5,0,0,"-"],reshaping:[5,0,0,"-"]},"hls4ml.converters.keras.convolution":{parse_conv1d_layer:[5,1,1,""],parse_conv2d_layer:[5,1,1,""]},"hls4ml.converters.keras.core":{BinaryQuantizer:[5,2,1,""],TernaryQuantizer:[5,2,1,""],parse_activation_layer:[5,1,1,""],parse_batchnorm_layer:[5,1,1,""],parse_dense_layer:[5,1,1,""],parse_input_layer:[5,1,1,""]},"hls4ml.converters.keras.graph":{parse_garnet_layer:[5,1,1,""]},"hls4ml.converters.keras.merge":{parse_merge_layer:[5,1,1,""]},"hls4ml.converters.keras.pooling":{parse_global_pooling_layer:[5,1,1,""],parse_pooling_layer:[5,1,1,""]},"hls4ml.converters.keras.reshape":{parse_conv2d_layer:[5,1,1,""],parse_permute_layer:[5,1,1,""],parse_reshape_layer:[5,1,1,""]},"hls4ml.converters.keras.reshaping":{parse_zeropadding1d_layer:[5,1,1,""],parse_zeropadding2d_layer:[5,1,1,""]},"hls4ml.converters.keras_to_hls":{KerasFileReader:[4,2,1,""],KerasModelReader:[4,2,1,""],compute_padding_1d:[4,1,1,""],compute_padding_2d:[4,1,1,""],get_qkeras_quantization:[4,1,1,""],get_supported_keras_layers:[4,1,1,""],keras_handler:[4,1,1,""],keras_to_hls:[4,1,1,""],parse_data_format:[4,1,1,""],parse_default_keras_layer:[4,1,1,""],register_keras_layer_handler:[4,1,1,""]},"hls4ml.converters.keras_to_hls.KerasFileReader":{get_weights_data:[4,3,1,""],get_weights_shape:[4,3,1,""]},"hls4ml.converters.keras_to_hls.KerasModelReader":{get_weights_data:[4,3,1,""],get_weights_shape:[4,3,1,""]},"hls4ml.converters.onnx":{convolution:[6,0,0,"-"],core:[6,0,0,"-"],merge:[6,0,0,"-"],pooling:[6,0,0,"-"],reshape:[6,0,0,"-"]},"hls4ml.converters.onnx.convolution":{parse_conv_layer:[6,1,1,""]},"hls4ml.converters.onnx.core":{parse_activation_layer:[6,1,1,""],parse_batchnorm_layer:[6,1,1,""],parse_gemm_layer:[6,1,1,""]},"hls4ml.converters.onnx.merge":{parse_merge_layer:[6,1,1,""]},"hls4ml.converters.onnx.pooling":{parse_global_pooling_layer:[6,1,1,""],parse_pool_layer:[6,1,1,""]},"hls4ml.converters.onnx.reshape":{parse_reshape_layer:[6,1,1,""],parse_transpose_layer:[6,1,1,""]},"hls4ml.converters.onnx_to_hls":{ONNXDataReader:[4,2,1,""],compute_pads_1d:[4,1,1,""],compute_pads_2d:[4,1,1,""],get_input_shape:[4,1,1,""],get_onnx_attribute:[4,1,1,""],get_onnx_input_name:[4,1,1,""],get_out_layer_name:[4,1,1,""],get_supported_onnx_layers:[4,1,1,""],onnx_handler:[4,1,1,""],onnx_to_hls:[4,1,1,""],register_onnx_layer_handler:[4,1,1,""],replace_char_inconsitency:[4,1,1,""],sanitize_layer_name:[4,1,1,""]},"hls4ml.converters.onnx_to_hls.ONNXDataReader":{add_input:[4,3,1,""],get_weights_data:[4,3,1,""]},"hls4ml.converters.utils":{compute_padding_1d:[4,1,1,""],compute_padding_2d:[4,1,1,""],parse_data_format:[4,1,1,""]},"hls4ml.model":{hls_layers:[8,0,0,"-"],hls_model:[8,0,0,"-"],optimizer:[9,0,0,"-"]},"hls4ml.model.hls_layers":{Activation:[8,2,1,""],ArrayVariable:[8,2,1,""],BatchNormalization:[8,2,1,""],BiasAdd:[8,2,1,""],CompressedType:[8,2,1,""],CompressedWeightVariable:[8,2,1,""],Concatenate:[8,2,1,""],Conv1D:[8,2,1,""],Conv2D:[8,2,1,""],Conv2DBatchnorm:[8,2,1,""],Dense:[8,2,1,""],DepthwiseConv2D:[8,2,1,""],Dot:[8,2,1,""],ExponentPrecisionType:[8,2,1,""],ExponentType:[8,2,1,""],ExponentWeightVariable:[8,2,1,""],FixedPrecisionType:[8,2,1,""],GarNet:[8,2,1,""],GarNetStack:[8,2,1,""],GlobalPooling1D:[8,2,1,""],GlobalPooling2D:[8,2,1,""],HLSType:[8,2,1,""],InplaceVariable:[8,2,1,""],Input:[8,2,1,""],IntegerPrecisionType:[8,2,1,""],Layer:[8,2,1,""],Merge:[8,2,1,""],PReLU:[8,2,1,""],PackedType:[8,2,1,""],ParametrizedActivation:[8,2,1,""],Pooling1D:[8,2,1,""],Pooling2D:[8,2,1,""],Quantizer:[8,2,1,""],Reshape:[8,2,1,""],Resize:[8,2,1,""],SeparableConv1D:[8,2,1,""],SeparableConv2D:[8,2,1,""],Softmax:[8,2,1,""],StreamVariable:[8,2,1,""],Transpose:[8,2,1,""],Variable:[8,2,1,""],WeightVariable:[8,2,1,""],XnorPrecisionType:[8,2,1,""],ZeroPadding1D:[8,2,1,""],ZeroPadding2D:[8,2,1,""],find_minimum_width:[8,1,1,""],register_layer:[8,1,1,""]},"hls4ml.model.hls_layers.Activation":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.ArrayVariable":{definition_cpp:[8,3,1,""],get_shape:[8,3,1,""],size:[8,3,1,""],size_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.BatchNormalization":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.BiasAdd":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.CompressedType":{definition_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.CompressedWeightVariable":{next:[8,3,1,""]},"hls4ml.model.hls_layers.Concatenate":{config_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Conv1D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Conv2D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Conv2DBatchnorm":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Dense":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.DepthwiseConv2D":{initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Dot":{config_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.ExponentType":{definition_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.ExponentWeightVariable":{next:[8,3,1,""]},"hls4ml.model.hls_layers.GarNet":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""],ref_impl:[8,4,1,""]},"hls4ml.model.hls_layers.GlobalPooling1D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.GlobalPooling2D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.HLSType":{definition_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.InplaceVariable":{definition_cpp:[8,3,1,""],get_shape:[8,3,1,""],size_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.Input":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Layer":{add_bias:[8,3,1,""],add_output_variable:[8,3,1,""],add_weights:[8,3,1,""],add_weights_variable:[8,3,1,""],config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],get_attr:[8,3,1,""],get_input_node:[8,3,1,""],get_input_variable:[8,3,1,""],get_layer_precision:[8,3,1,""],get_numbers_cpp:[8,3,1,""],get_output_nodes:[8,3,1,""],get_output_variable:[8,3,1,""],get_variables:[8,3,1,""],get_weights:[8,3,1,""],initialize:[8,3,1,""],make_array_variable:[8,3,1,""],make_stream_variable:[8,3,1,""],precision_cpp:[8,3,1,""],set_attr:[8,3,1,""]},"hls4ml.model.hls_layers.Merge":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.PReLU":{function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.PackedType":{definition_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.ParametrizedActivation":{function_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.Pooling1D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Pooling2D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Reshape":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Resize":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.SeparableConv1D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.SeparableConv2D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.Softmax":{initialize:[8,3,1,""]},"hls4ml.model.hls_layers.StreamVariable":{get_shape:[8,3,1,""],size:[8,3,1,""],size_cpp:[8,3,1,""]},"hls4ml.model.hls_layers.Transpose":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.WeightVariable":{definition_cpp:[8,3,1,""],next:[8,3,1,""],update_precision:[8,3,1,""]},"hls4ml.model.hls_layers.ZeroPadding1D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_layers.ZeroPadding2D":{config_cpp:[8,3,1,""],function_cpp:[8,3,1,""],initialize:[8,3,1,""]},"hls4ml.model.hls_model":{HLSConfig:[8,2,1,""],HLSModel:[8,2,1,""]},"hls4ml.model.hls_model.HLSConfig":{get_bram_size:[8,3,1,""],get_compression:[8,3,1,""],get_config_value:[8,3,1,""],get_conv_implementation:[8,3,1,""],get_layer_config:[8,3,1,""],get_layer_config_value:[8,3,1,""],get_output_dir:[8,3,1,""],get_precision:[8,3,1,""],get_project_name:[8,3,1,""],get_reuse_factor:[8,3,1,""],get_strategy:[8,3,1,""],get_target_cycles:[8,3,1,""],is_resource_strategy:[8,3,1,""]},"hls4ml.model.hls_model.HLSModel":{build:[8,3,1,""],compile:[8,3,1,""],get_bram_variables:[8,3,1,""],get_input_variables:[8,3,1,""],get_layer_output_variable:[8,3,1,""],get_layers:[8,3,1,""],get_output_variables:[8,3,1,""],get_weights_data:[8,3,1,""],insert_node:[8,3,1,""],make_node:[8,3,1,""],next_layer:[8,3,1,""],predict:[8,3,1,""],register_bram_variable:[8,3,1,""],register_output_variable:[8,3,1,""],remove_node:[8,3,1,""],replace_node:[8,3,1,""],trace:[8,3,1,""],write:[8,3,1,""]},"hls4ml.model.optimizer":{optimizer:[9,0,0,"-"],passes:[10,0,0,"-"]},"hls4ml.model.optimizer.optimizer":{OptimizerPass:[9,2,1,""],get_available_passes:[9,1,1,""],get_optimizer:[9,1,1,""],optimize_model:[9,1,1,""],register_pass:[9,1,1,""]},"hls4ml.model.optimizer.optimizer.OptimizerPass":{match:[9,3,1,""],transform:[9,3,1,""]},"hls4ml.model.optimizer.passes":{bn_fuse:[10,0,0,"-"],bn_quant:[10,0,0,"-"],clone:[10,0,0,"-"],conv_same_pad:[10,0,0,"-"],fuse_biasadd:[10,0,0,"-"],multi_dense:[10,0,0,"-"],nop:[10,0,0,"-"],pointwise:[10,0,0,"-"],repack_stream:[10,0,0,"-"],transpose_opt:[10,0,0,"-"]},"hls4ml.model.optimizer.passes.bn_fuse":{FuseBatchNormalization:[10,2,1,""]},"hls4ml.model.optimizer.passes.bn_fuse.FuseBatchNormalization":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.bn_quant":{BatchNormalizationQuantizedTanh:[10,2,1,""],MergeBatchNormAndQuantizedTanh:[10,2,1,""],QuantizeDenseOutput:[10,2,1,""]},"hls4ml.model.optimizer.passes.bn_quant.BatchNormalizationQuantizedTanh":{config_cpp:[10,3,1,""],function_cpp:[10,3,1,""],initialize:[10,3,1,""]},"hls4ml.model.optimizer.passes.bn_quant.MergeBatchNormAndQuantizedTanh":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.bn_quant.QuantizeDenseOutput":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.clone":{Clone:[10,2,1,""],CloneOutput:[10,2,1,""]},"hls4ml.model.optimizer.passes.clone.Clone":{config_cpp:[10,3,1,""],function_cpp:[10,3,1,""],initialize:[10,3,1,""]},"hls4ml.model.optimizer.passes.clone.CloneOutput":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.conv_same_pad":{InsertZeroPaddingBeforeConv1D:[10,2,1,""],InsertZeroPaddingBeforeConv2D:[10,2,1,""]},"hls4ml.model.optimizer.passes.conv_same_pad.InsertZeroPaddingBeforeConv1D":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.conv_same_pad.InsertZeroPaddingBeforeConv2D":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.fuse_biasadd":{FuseBiasAdd:[10,2,1,""]},"hls4ml.model.optimizer.passes.fuse_biasadd.FuseBiasAdd":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.multi_dense":{ReplaceMultidimensionalDenseWithConv:[10,2,1,""]},"hls4ml.model.optimizer.passes.multi_dense.ReplaceMultidimensionalDenseWithConv":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.nop":{EliminateLinearActivation:[10,2,1,""]},"hls4ml.model.optimizer.passes.nop.EliminateLinearActivation":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.pointwise":{OptimizePointwiseConv:[10,2,1,""],PointwiseConv1D:[10,2,1,""],PointwiseConv2D:[10,2,1,""]},"hls4ml.model.optimizer.passes.pointwise.OptimizePointwiseConv":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.repack_stream":{Broadcast:[10,2,1,""],BroadcastStream:[10,2,1,""],Repack:[10,2,1,""],ReshapeStream:[10,2,1,""]},"hls4ml.model.optimizer.passes.repack_stream.Broadcast":{config_cpp:[10,3,1,""],function_cpp:[10,3,1,""],initialize:[10,3,1,""]},"hls4ml.model.optimizer.passes.repack_stream.BroadcastStream":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.repack_stream.Repack":{config_cpp:[10,3,1,""],function_cpp:[10,3,1,""],initialize:[10,3,1,""]},"hls4ml.model.optimizer.passes.repack_stream.ReshapeStream":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.model.optimizer.passes.transpose_opt":{RemoveUselessTranspose:[10,2,1,""]},"hls4ml.model.optimizer.passes.transpose_opt.RemoveUselessTranspose":{match:[10,3,1,""],transform:[10,3,1,""]},"hls4ml.report":{vivado_report:[11,0,0,"-"]},"hls4ml.report.vivado_report":{parse_vivado_report:[11,1,1,""],read_vivado_report:[11,1,1,""]},"hls4ml.templates":{templates:[12,0,0,"-"],vivado_template:[12,0,0,"-"]},"hls4ml.templates.templates":{Backend:[12,2,1,""],get_backend:[12,1,1,""],register_backend:[12,1,1,""]},"hls4ml.templates.templates.Backend":{get_config_template:[12,3,1,""],get_function_template:[12,3,1,""],get_include_list:[12,3,1,""],register_source:[12,3,1,""],register_templates:[12,3,1,""]},"hls4ml.templates.vivado_template":{VivadoBackend:[12,2,1,""]},"hls4ml.templates.vivado_template.VivadoBackend":{compute_conv1d_instructions:[12,3,1,""],compute_conv2d_instructions:[12,3,1,""],convert_precision_string:[12,3,1,""],get_closest_reuse_factor:[12,3,1,""],get_valid_reuse_factors:[12,3,1,""],product_type:[12,3,1,""],set_closest_reuse_factor:[12,3,1,""],set_target_reuse_factor:[12,3,1,""]},"hls4ml.utils":{config:[13,0,0,"-"],example_models:[13,0,0,"-"],plot:[13,0,0,"-"]},"hls4ml.utils.config":{config_from_keras_model:[13,1,1,""],config_from_onnx_model:[13,1,1,""],config_from_pytorch_model:[13,1,1,""],create_vivado_config:[13,1,1,""]},"hls4ml.utils.example_models":{fetch_example_list:[13,1,1,""],fetch_example_model:[13,1,1,""]},"hls4ml.utils.plot":{add_edge:[13,1,1,""],check_pydot:[13,1,1,""],model_to_dot:[13,1,1,""],plot_model:[13,1,1,""]},"hls4ml.writer":{vivado_writer:[14,0,0,"-"],writers:[14,0,0,"-"]},"hls4ml.writer.vivado_writer":{VivadoWriter:[14,2,1,""]},"hls4ml.writer.vivado_writer.VivadoWriter":{print_array_to_cpp:[14,3,1,""],type_definition_cpp:[14,3,1,""],variable_definition_cpp:[14,3,1,""],write_bridge:[14,3,1,""],write_build_script:[14,3,1,""],write_defines:[14,3,1,""],write_hls:[14,3,1,""],write_nnet_utils:[14,3,1,""],write_parameters:[14,3,1,""],write_project_cpp:[14,3,1,""],write_project_dir:[14,3,1,""],write_project_header:[14,3,1,""],write_tar:[14,3,1,""],write_test_bench:[14,3,1,""],write_weights:[14,3,1,""],write_yml:[14,3,1,""]},"hls4ml.writer.writers":{Writer:[14,2,1,""],get_writer:[14,1,1,""],register_writer:[14,1,1,""]},"hls4ml.writer.writers.Writer":{write_hls:[14,3,1,""]},hls4ml:{converters:[4,0,0,"-"],model:[8,0,0,"-"],report:[11,0,0,"-"],templates:[12,0,0,"-"],utils:[13,0,0,"-"],writer:[14,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"1x1":10,"case":[0,2,8,16,17],"class":[4,5,8,9,10,12,14],"const":0,"default":[0,2,4,8,13,19,20],"export":[8,15,19,20],"final":16,"float":[2,16],"function":[1,4,8,12,13,16,17,19,20],"import":[0,1,2,4,13,16,20],"int":[4,13],"new":[8,16,17,19,20,21],"return":[1,4,8,12,13,20],"static":0,"true":[0,4,8,13,14],"var":[8,14],"while":[0,13,16],Adding:19,And:20,For:[0,15,16,17,20],HLS:[4,13,15,16,17,19,20,21],NNs:21,One:[0,4],The:[0,1,2,4,8,10,13,17,20,21],Then:[0,20],There:[0,20],Use:[13,20],Using:[0,2,16],With:16,_out:8,about:[15,20],abov:[0,13,16],abs:21,acat:18,accept:16,access:19,accum_t:0,accuraci:2,achiev:[2,16],activ:[0,2,8,10,16,19],add:8,add_bia:8,add_edg:13,add_input:4,add_output_vari:8,add_weight:8,add_weights_vari:8,added:[8,17,19],addit:[0,4,20],addition:[0,13],advanc:[0,1],advis:13,affect:13,after:[0,1,10,16,19,20],agglomer:16,algorithm:[0,4,16,17,20],all:[0,1,8,13,15,16,19],allow:[2,13,16,19],along:20,alpha:19,alreadi:[1,16],also:[0,1,13,16,18,20,21],altern:20,ambigu:8,analysi:16,ani:[0,15,19],anyth:10,ap_fix:[0,4,12,13],ap_int:[0,13],ap_uint:0,apart:20,api:[1,4,19,20],appear:0,appli:13,applic:16,appropri:[1,2],architectur:[0,16,21],arg:4,argument:[2,4,13,15],arithmet:16,aritifici:0,arrai:[1,4,8],array_partit:0,arrayvari:8,arxiv:[18,19,21],as_refer:14,aspect:16,aster:19,attempt:8,attribut:[8,10],atyp:[8,14],auto:8,automat:[0,16],avail:[2,13],avoid:2,awar:[16,19],backend:[12,19],backend_cl:12,balanc:16,bare:2,bartsia:19,base:[4,5,8,9,10,12,14],basic:[0,20],batch:[10,19],batchnorm:[8,19],batchnormalization1:0,batchnormalizationquantizedtanh:10,bdt:21,been:[16,21],befor:[1,8,16],beginn:20,below:[0,2,15,21],benefit:16,benjamin:18,best:16,beta:[10,19],better:0,between:[10,16],bia:[0,16],bias:[0,13],bias_t:0,biasadd:[8,10],bin:16,binari:[0,10,13,18,19],binaryquant:5,bit:[0,2,5,8,13,16],bn_fuse:[3,8,9],bn_quant:[3,8,9],bnn:8,bool:[0,8],boost:18,both:[0,2],box:[2,20],boxplot:2,bring:21,broadcast:10,broadcaststream:10,bug:19,bugfix:19,build:[8,16,19,20],build_prj:20,built:[19,20],calcul:16,california:18,call:[2,4,15,16],calucl:0,can:[0,1,2,8,13,15,16,17,19,20],caus:4,cern:18,certain:16,champaign:18,chanc:0,chang:[0,13,19],channels_last:4,chapter:[0,20],charact:4,check:20,check_pydot:13,checkout:0,chep:18,chicago:18,chieh:18,choos:0,chosen:2,chosen_rf:12,cite:18,clazz:8,clock:[0,4],clock_period:[4,13],clockperiod:[0,4],clone:[3,8,9,20],cloneoutput:10,close:12,closest:12,cluster:13,cnn:[19,21],code:[2,4,16,20,21],colleg:18,collid:16,collis:16,columbia:18,com:[13,17,20],combin:2,come:16,command:[1,19],common:10,commun:16,compactli:16,compar:[2,16],compil:[8,19],complet:0,complex:16,compress:[0,8,16,18],compressedtyp:8,compressedweightvari:8,comput:[0,16],compute_conv1d_instruct:12,compute_conv2d_instruct:12,compute_padding_1d:4,compute_padding_2d:4,compute_pads_1d:4,compute_pads_2d:4,concaten:[8,19],concept:[0,20],config2:0,config4:0,config:[0,1,2,3,4,5,6,8,19,20],config_cpp:[8,10],config_fil:4,config_from_keras_model:[0,1,13],config_from_onnx_model:[4,13],config_from_pytorch_model:[4,13],config_templ:12,configur:[1,2,4,13,15,16,17,19,20],conif:21,connect:[8,18,21],consid:[0,4,16],construct:[0,8],contact:17,contain:[2,4,16],containig:4,control:[16,20],conv1d:[8,10,19,21],conv2d:[8,10,21],conv2dbatchnorm:8,conv_same_pad:[3,8,9],conveni:8,convers:[4,13,15,16,20],convert:[1,2,3,12,13,16,19,20],convert_from_config:4,convert_from_keras_model:[1,4,13],convert_from_onnx_model:[4,13],convert_from_pytorch_model:[4,13],convert_precision_str:12,convolut:[3,4,19,21],core:[3,4],correct:[8,19],correspond:13,cosim:[8,20],cosimul:19,cost:16,cours:18,cover:2,cpp:0,cpu:[2,16],creat:[0,4,8,13,15,16,17,19,20],create_vivado_config:13,creation:16,csim:[1,8,20],csimul:15,current:[0,15,17],custom:[13,19],dat:0,data:[0,2,4,8,13,16,18],data_format:4,data_read:[5,8],data_t:12,dataflow:16,debug:1,decid:2,decis:[16,18],dedic:20,deep:18,default_precis:13,default_reuse_factor:13,defin:[0,4,13,16,20],definition_cpp:8,denot:[0,13],dens:[0,8,10,19],dense1:0,dense2:0,dense_config:0,dense_lat:0,depend:[2,16,21],deprec:19,depth:8,depthwiseconv2d:8,describ:[0,15],design:[4,16],destination_dir:12,detail:[15,16,17,20],detector:16,determin:[12,16],develop:[0,17,21],devic:4,diagram:2,dict:[4,8,13],dictionari:[0,1,4,13,19],diego:18,differ:[0,1,2,10,16,19,20],differenti:8,dim:0,dim_nam:8,directori:[0,4,13,15,19,20],disabl:20,discuss:20,displai:13,distribut:2,dnn:[0,19],doc:4,document:[0,1,15,19,20,21],doe:8,doesn:10,doing:[1,15],done:[0,16,20],dot:[8,13,18],download:[13,20],dpi:13,dst:13,duart:18,duc:18,dure:[4,12,16],dylan:18,dynam:16,each:[0,1,2,16,20],edward:18,effici:16,either:0,eliminatelinearactiv:10,emul:19,enabl:[13,16,19],energi:16,enhanc:19,ensur:[4,16],entir:13,entri:8,environ:16,equal:12,error:19,escienc:18,especi:19,essenti:16,establish:2,etc:12,evalu:[2,16],even:[0,16],event:16,everi:[0,13],everyth:16,exampl:[0,4,8,13,15,19],example_model:[3,20],except:[4,8,13],exist:[8,19],exit:15,expens:16,experi:17,explicitli:16,explor:20,expon:8,exponentprecisiontyp:8,exponenttyp:8,exponentweightvari:8,express:[8,16],extend:2,extra:[2,20],extract:4,extrem:16,factor:[0,4,10,13,16,19],fals:[0,8,11,13,14,19],familiar:20,fast:[16,18],faster:1,fastmachinelearn:20,fc1:0,featur:[17,20],feel:20,fermilab:18,fetch:20,fetch_example_list:[13,20],fetch_example_model:[13,20],field:0,figur:[2,16],file:[2,4,13,15,19,20],file_nam:12,filesystem:4,filt_height:4,filt_siz:4,filt_width:4,filter:16,find:[8,20],find_minimum_width:8,fine:[0,2,13],finer:0,firmwar:[0,17],first:[16,19,20],fix:[0,13,16,19],fixedprecisiontyp:8,flip:10,flow:19,flvb2104:[0,4,13],fold:10,follow:[8,15,16,20],format:[0,2,13],found:[0,17,20],four:16,fpga:[0,2,4,13,15,16,17,18],fpga_part:[4,13],framework:19,freedom:16,from:[0,1,2,4,8,12,13,16,19,20],full:[15,19,20],full_report:11,fulli:21,function_cpp:[8,10],function_templ:12,further:[0,2,19],fuse:[10,19],fuse_biasadd:[3,8,9],fuse_consecutive_batch_norm:19,fusebatchnorm:10,fusebiasadd:10,g_m:16,gamma:10,garnet:[8,19],garnetstack:8,gather:19,gener:[0,1,4,13,15,16,19,20],get:[1,4,15,16],get_attr:8,get_available_pass:9,get_backend:12,get_bram_s:8,get_bram_vari:8,get_closest_reuse_factor:12,get_compress:8,get_config_templ:12,get_config_valu:8,get_conv_implement:8,get_function_templ:12,get_include_list:12,get_input_nod:8,get_input_shap:4,get_input_vari:8,get_lay:8,get_layer_config:8,get_layer_config_valu:8,get_layer_output_vari:8,get_layer_precis:8,get_numbers_cpp:8,get_onnx_attribut:4,get_onnx_input_nam:4,get_optim:9,get_out_layer_nam:4,get_output_dir:8,get_output_nod:8,get_output_vari:8,get_precis:8,get_project_nam:8,get_qkeras_quant:4,get_reuse_factor:8,get_shap:8,get_strategi:8,get_supported_keras_lay:4,get_supported_onnx_lay:4,get_target_cycl:8,get_valid_reuse_factor:12,get_vari:8,get_weight:8,get_weights_data:[4,8],get_weights_shap:4,get_writ:14,get_ymodel_kera:1,git:20,github:[13,17,19,20,21],giusepp:18,give:[2,16,20],given:[2,13,16,21],globalpooling1d:8,globalpooling2d:8,gmail:17,goal:16,going:21,good:2,gor:19,gpu:16,granular:[0,1,2,4,13,19],granulr:13,graph:[3,4,6,8],graphviz:13,greatli:16,grei:2,guglielmo:18,guid:20,h5py:20,had:16,hadron:16,han:18,hand:18,handl:19,handler_func:4,hardwar:16,harri:18,has:[0,13,16,21],have:[0,1,15,21],hawkeye360:18,head:0,help:[1,2,15,16,17],helper:[8,12],here:[0,1,19,20],hierachi:0,high:[16,17],higher:19,highest:16,highli:[13,16],histogram:2,hl4ml:0,hls4ml:[0,1,2,18,19,21],hls4ml_prj:1,hls:[0,1,4,13,15,17,20],hls_config:[1,4,13,19],hls_dir:11,hls_layer:[3,5,10],hls_model:[1,2,3,4,13,19,20],hls_type:8,hlsconfig:[0,4,8],hlsmodel:[2,4,8],hlstype:8,hoang:18,horizont:13,how:[0,13,15,17,20],howev:[4,16],hsu:18,http:[13,17,20,21],illinoi:18,imag:13,implement:[0,4,10,12,16,17,18,19,20,21],impli:15,implment:20,importerror:13,improv:16,in_c:12,in_h:12,in_height:4,in_siz:4,in_w:12,in_width:4,inappropri:2,inch:13,includ:[17,20,21],include_list:12,inconsist:4,independ:16,index:[4,8],index_precis:8,index_t:0,indic:2,individu:1,infer:[12,15,16,17,18],inform:[0,4,13,16,20],initi:[0,8,10,13,16,19],initialis:2,inplacevari:8,input:[0,1,2,4,8,10,13,16,19],input_1:0,input_idx:4,input_nam:[4,5,8],input_shap:[4,5,6],input_t:0,inputdata:0,inputs_map:6,insert:[0,8,10],insert_nod:8,insertzeropaddingbeforeconv1d:10,insertzeropaddingbeforeconv2d:10,inspect:13,instal:[2,13,19],instanc:13,instead:[0,4],instruct:20,integ:[0,2,8,13],integerprecisiontyp:[8,12],interest:17,interfac:[1,20],intern:12,interv:[0,16,19],intuit:16,invest:16,io_parallel:[0,4,13],io_seri:[0,4,19],io_stream:[4,19,21],io_typ:[0,4,13],iotyp:[0,4,19,21],is_resource_strategi:8,isn:16,issu:[4,19],its:[0,1],javier:18,jennif:18,jindariani:18,jinst:[16,18],json:[0,13,15,19,20],jupyt:13,just:[0,1,15],kei:[1,8,13],kept:16,kera:[0,1,2,3,4,13,15,16,19,20,21],keras_3lay:[0,13,20],keras_3layer_input_featur:0,keras_3layer_predict:0,keras_3layer_weight:0,keras_handl:4,keras_lay:[4,5],keras_model:[1,2,4],keras_to_hl:[2,3,20],keras_trac:1,kerasfileread:4,kerash5:[0,4],kerasjson:0,kerasmodelread:4,kernel:[10,16],kernel_s:12,keyword:2,kind:[8,12],know:17,krei:18,kreinar:18,kwarg:8,languag:[16,17],larg:16,larger:[0,19,21],largest:16,last:20,latenc:[0,16,19],latest:[17,21],layer2_out:0,layer2_t:0,layer3_out:0,layer3_t:0,layer4_out:0,layer4_t:0,layer5_out:0,layer:[1,4,8,10,12,13,16,19,21],layer_list:8,layer_nam:[4,8],layernam:0,layertyp:0,leaf:8,learn:[0,13,15,16,17,20],least:2,left:2,let:[0,16,17],level:[16,17],lhc:16,librari:[0,19],licens:20,like:[0,4,15,16,17],line:[1,13,19,20],link:20,linux:21,list:[1,8,20,21],live:16,load:2,load_data:2,load_model:2,locat:4,logic:19,loncar:18,longer:16,look:[0,4,16],loss:2,low:[2,16],lower:16,lowest:16,lstm:21,machin:[0,13,15,16,17,20],maco:21,made:8,mai:[0,2,4],main:20,make:[2,8],make_array_vari:8,make_nod:8,make_stream_vari:8,mani:16,mark:18,match:[9,10],mathbf:16,matplotlib:2,matrix:16,maurizio:18,max:19,maxim:16,mean:[0,10,16],median:2,merg:[3,4,8,10,19],mergebatchnormandquantizedtanh:10,messag:15,messeg:15,method:[0,2],microsecond:16,might:[0,2,20],min:19,mind:16,minim:4,minimum:8,minor:19,minut:20,mit:18,mix:19,mlp:21,mode:19,model:[0,2,3,4,5,13,14,15,16,17,19,20],model_default_t:0,model_nam:13,model_to_dot:13,modul:2,more:[0,2,4,10,13,15,16,20],most:[0,16],move:[19,21],much:[1,16,19],multi:[16,21],multi_dens:[3,8,9],multipl:[8,10,16,19],multipli:16,must:[0,16,20],my_keras_model:4,myproject:[0,4,13],n_elem:8,n_in:0,n_input_1_1:0,n_layer_2:0,n_layer_4:0,n_m:16,n_nonzero:0,n_out:0,n_pack:8,n_zero:0,name:[0,1,4,8,9,10,12,13,14,15,19],name_suffix:14,necessari:16,need:[2,16,19,20],nest:13,network:[0,2,16,18,21],neubauer:18,neural:[0,2,16,18,21],neuron:16,new_nod:8,new_precis:8,next:[0,8,20],next_lay:8,ngadiuba:18,nhan:18,nnet:0,nnet_util:12,noah:18,node:[4,6,8,9,10],non:2,none:[0,2,4,8,9,10],nontrivi:16,nop:[3,8,9],normal:[0,2,10],normalis:19,note:[0,1,4,8,16,20,21],notebook:13,now:[4,19],npy:0,number:[0,4,12,13,15,16],numer:2,numpi:[1,4,20],object:[1,2,4,8,9,12,13,14,19],obtain:[0,12,19],odir:14,offer:16,offici:19,offlin:16,often:[15,16],old_nod:8,onc:10,one:[0,8,16,19],ones:8,onli:[0,2,4,16],onnx:[3,4,13,15,20,21],onnx_handl:4,onnx_to_hl:3,onnxdataread:4,open:[2,16,17],oper:[4,19],opportun:0,opt_cl:9,optim:[0,3,8,16,19],optimize_model:9,optimizepointwiseconv:10,optimizerpass:[9,10],optims:19,option:[0,2,4,8,13,15,16,19],order:[16,20],org:[20,21],orient:0,other:[0,20,21],our:[1,16,20],out:[19,20],out_nam:8,outpu:8,output:[0,1,4,8,10,13,15,16,19],output_dir:[1,4,13],output_nam:8,outputdir:[0,4],outputpredict:0,overflow:[2,8],own:0,p07027:[16,18],pack:10,packag:[16,17,18,21],packedtyp:8,pad:12,pad_typ:4,page:[0,1,15,16,17,20,21],paladino:18,paper:16,parallel:[16,19],paramet:[0,4,8,10,13,16,20],parametrizedactiv:8,pars:[4,16,20],parse_activation_lay:[5,6],parse_batchnorm_lay:[5,6],parse_conv1d_lay:5,parse_conv2d_lay:5,parse_conv_lay:6,parse_data_format:4,parse_default_keras_lay:4,parse_dense_lay:5,parse_garnet_lay:5,parse_gemm_lay:6,parse_global_pooling_lay:[5,6],parse_input_lay:5,parse_merge_lay:[5,6],parse_permute_lay:5,parse_pool_lay:6,parse_pooling_lay:5,parse_reshape_lay:[5,6],parse_transpose_lay:6,parse_vivado_report:11,parse_yaml_config:4,parse_zeropadding1d_lay:5,parse_zeropadding2d_lay:5,part:[0,4,13,16],particl:18,particular:[0,4,15,20],partit:8,pass:[3,4,8,9,13],path:[0,4],per:[13,19],perceptron:21,perform:[2,16],period:[0,4],perm:[4,10],philip:18,physic:[16,18],pierini:18,pip:[2,20],pipelin:[0,4,16,19],plan:17,pleas:[1,4,17,18,20],plot:[2,3],plot_model:13,plt:2,plug:16,png:13,po2:[8,19],point:[0,2,13,16],pointwis:[3,8,9],pointwiseconv1d:10,pointwiseconv2d:10,pool:[3,4],pooling1d:8,pooling2d:8,posit:15,possibl:[0,2,16,19],potenti:16,power:16,pragma:[0,8],precis:[0,2,4,8,12,13,16,18,19],precision_cpp:8,precomput:16,precsion:[0,13],predict:[0,2,8,16,19],predict_ouput:1,prefer:1,prelu:8,present:4,preserv:16,previou:[8,15,16],previous:19,princip:16,print:[19,20],print_array_to_cpp:14,prj:20,problem:16,process:16,produc:[0,2,16],product:12,product_typ:12,profil:[1,3,19,20],program:[15,16],project:[0,1,4,15,16,19,20],project_nam:[4,13],projectnam:[0,4],proper:4,protobuf:[19,20],prototyp:[1,16,21],proven:16,provid:[0,2,4,15,16,20,21],proxi:8,pseudo:2,pydot:13,pypi:[19,20],pyplot:2,python:[2,17,19,20],pytorch:[3,4,13,15,16,19,20,21],pytorch_to_hl:3,pyyaml:20,qkera:[3,4,8,9,16,19,21],qkeras_lay:[3,4],quantiz:[5,8,10,16,19],quantizedenseoutput:10,quartil:2,quickli:[0,1,20],rais:[4,8,13],rang:2,rankdir:13,rankin:18,rapid:16,read:[1,20],read_vivado_report:[1,11,20],reader:[4,6],real:2,realiz:16,realli:[0,4,16],recurr:21,redefinit:19,reduc:[2,16],ref_impl:8,refer:[0,1,4,16,19],refin:19,regist:8,register_backend:12,register_bram_vari:8,register_keras_layer_handl:4,register_lay:8,register_onnx_layer_handl:4,register_output_vari:8,register_pass:9,register_sourc:12,register_templ:12,register_writ:14,regular:8,rel:16,relat:[13,19],releas:21,relev:4,reload:19,relu:0,relu_config3:0,remap:16,rememb:0,remov:[8,10,15],remove_nod:8,removeuselesstranspos:10,repack:10,repack_stream:[3,8,9],replac:[4,8],replace_char_inconsit:4,replace_nod:8,replacemultidimensionaldensewithconv:10,repo:[13,19],report:[1,3,19,20],repositori:[20,21],repres:[0,2,8,13,16],request:20,requir:[0,8,16,20],research:18,reset:[8,15],reshap:[3,4,8,10],reshapestream:10,resiz:8,resourc:[0,2,16,19,20],respect:16,respons:0,result:[0,1,2,16],result_t:0,retriev:19,reus:[0,4,13,16,19],reuse_factor:[0,8],reusefactor:[0,2,4],rewir:8,rhode:18,right:2,rivera:18,rnn:21,rounding_mod:8,rtl:[15,19,20],run:[0,2,4,15,16,19,20],rutger:18,ryan:18,safe:2,said:16,same:8,san:18,sanitize_layer_nam:4,satisfactori:2,satur:8,saturation_bit:8,saturation_mod:8,save:[0,13,16,19],scale:[16,19],scienc:18,script:19,section:[0,1,15,20],see:[0,13,17,19,20,21],select:16,seminar:18,separ:[13,19],separableconv1d:8,separableconv2d:8,sequenc:[8,16],sequenti:[4,16],sergo:18,serial:[4,16,19],serv:13,set:[0,1,8,13,16,20],set_attr:8,set_closest_reuse_factor:12,set_target_reuse_factor:12,setup:[0,15,16,21],sever:[1,20,21],shade:2,shape:[8,13,19],shift:19,shih:18,should:[0,4,8,16],show:[2,15,20],show_layer_nam:13,show_precis:13,show_shap:13,shown:[2,16],sigmoid:0,sigmoid_config5:0,sign:[0,8,10,13,16],significantli:16,similar:1,simpl:[0,1],simpli:20,simplifi:16,simul:[0,1,2,15,19],singl:16,sioni:18,sizabl:16,size:[0,8,13,16],size_cpp:8,skipoptim:19,smallest:12,snippet:0,softmax:[8,19],softwar:20,some:[0,2,4,19,20],song:18,sort:12,sourc:[12,16,17],specif:[0,13,16],specifi:[0,2,8,13,19],speed:16,speedup:16,split:16,src:13,stabl:21,stage:0,standalon:19,start:[2,15],stem:16,step:[13,15,20],still:2,store:[0,13],store_weights_in_bram:0,str:[4,8,13],strategi:[0,19],stream:[10,19],streamvari:8,stride:[4,12],stride_height:4,stride_width:4,string:[4,12,13],struct:0,style:2,subgraph:13,submodul:3,subtract:19,suffici:2,suggest:0,suit:16,sum:16,summari:21,summer:18,superced:19,suppli:[0,1],support:[0,1,4,13,15,19,20,21],suppos:[1,15],suppport:21,sure:2,synth:[8,20],synthes:20,synthesi:[15,16,17,19,20],tabl:21,take:[16,20],taken:16,tanh:10,target:[0,4],task:16,tcl:20,techniqu:18,templat:3,tensorflow:[15,19,21],ternari:[10,18,19],ternaryquant:5,test:[0,2,4,13,15,20,21],test_prj:1,tf_to_hl:[3,19],than:[0,10,19],thei:0,them:[0,1,19],therebi:16,therefor:16,thi:[0,1,4,8,13,15,16,20,21],those:[0,8],though:16,three:2,threshold:10,through:[0,1,4,19],throughout:0,throughput:16,thu:16,time:[10,16],to_fil:13,too:16,tool:[2,16,19,20,21],toolbox:20,toolkit:21,torch:20,total:[0,13],trace:[8,19],trace_output:1,tradit:17,train:[0,2,16,19,20],tran:18,transform:[9,10],translat:[16,17,19,20],transpil:16,transpos:[4,8,10],transpose_opt:[3,8,9],tree:18,trick:16,trigger:16,tune:13,tutori:20,tweak:13,twepp:18,two:[0,2,12],type:[0,2,4,8,13,15,19],type_definition_cpp:14,type_nam:8,typedef:0,typic:16,under:0,understand:[0,20],univers:18,unsign:0,update_precis:8,urbana:18,usag:[0,1,2,15],use:[0,1,2,12,13,15,16,17,20],use_bia:19,used:[0,1,2,4,8,10,13,16,19],useful:[1,16],user:[0,13,16,20],uses:[0,2,16],using:[0,1,2,16,17,18,19,20],util:[0,1,3,16,19,20],valid:[2,4,8,15,19],valid_rf:12,valu:[1,2,8,12,16],var_nam:[4,8],vari:16,variabl:[0,2,4,8,13],variable_definition_cpp:14,varianc:10,variou:[17,20],vector:16,verbos:13,veri:1,version:[1,15,21],vertic:13,violinplot:2,virtex:0,visit:[0,15,20],visual:13,viti:21,vivado:[15,20,21],vivado_hl:20,vivado_report:3,vivado_synthesi:15,vivado_templ:3,vivado_writ:3,vivadobackend:12,vivadowrit:14,vladimir:18,vsynth:8,wai:0,want:[0,1,13,20],warn:19,washington:18,weight:[0,2,4,13,15,16,19,20],weight_t:[0,12],weightvari:8,welcom:0,what:[0,19],when:[0,1,2,4,8],where:[0,1,4,13],whether:[2,13,16],which:[0,2,4,8,10,12,16],whisker:2,whole:[13,15],whose:10,width:[8,16],window:21,within:0,without:[0,8,19],work:[0,13,20],workflow:20,would:[2,16,17],write:[4,8,15,19],write_bridg:14,write_build_script:14,write_defin:14,write_hl:14,write_nnet_util:14,write_paramet:14,write_project_cpp:14,write_project_dir:14,write_project_head:14,write_tar:14,write_test_bench:14,write_txt_fil:14,write_weight:14,write_yml:14,writer:3,writer_cl:14,www:20,xcku115:[0,4,13],xilinx:[0,20],xilinxpart:[0,4],xnor:8,xnorprecisiontyp:8,yaml:[2,4],yet:[19,21],yml:[0,2,15,19,20],ymlfile:2,you:[0,1,2,4,13,15,17,18,19,20],your:[0,1,2,4,13,15,16,17,20],zero:2,zeropadding1d:8,zeropadding2d:8,zhenbin:18,zurich:18},titles:["Configuration","HLS Model Class","Profiling","hls4ml package","hls4ml.converters package","hls4ml.converters.keras package","hls4ml.converters.onnx package","hls4ml.converters.pytorch package","hls4ml.model package","hls4ml.model.optimizer package","hls4ml.model.optimizer.passes package","hls4ml.report package","hls4ml.templates package","hls4ml.utils package","hls4ml.writer package","Command Line Interface","Concepts","Welcome to hls4ml\u2019s documentation!","Reference and Contributors","Release Notes","Setup","Status and Features"],titleterms:{"class":1,HLS:[0,1],The:16,addit:18,api:0,bn_fuse:10,bn_quant:10,build:[1,15],citat:18,clone:10,code:0,command:[15,20],compil:1,concept:16,config:[13,15],configur:0,content:[3,4,5,6,7,8,9,10,11,12,13,14],contributor:18,conv_same_pad:10,convert:[0,4,5,6,7,15],convolut:[5,6,7],core:[5,6,7],depend:20,detail:0,document:17,exampl:[20,21],example_model:13,exist:20,featur:21,file:0,further:20,fuse_biasadd:10,get:20,graph:5,help:20,hls4ml:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20],hls_layer:8,hls_model:8,how:16,inspir:16,instal:20,interfac:15,kera:5,keras_to_hl:4,layer:0,level:0,line:15,merg:[5,6],method:1,model:[1,8,9,10,21],modul:[3,4,5,6,7,8,9,10,11,12,13,14],multi_dens:10,nop:10,note:19,onnx:6,onnx_to_hl:4,optim:[9,10],option:20,overview:15,packag:[3,4,5,6,7,8,9,10,11,12,13,14],pass:10,per:0,plot:13,pointwis:10,pool:[5,6,7],predict:1,present:18,profil:[2,8],project:17,python:0,pytorch:7,pytorch_to_hl:4,qkera:[5,10],qkeras_lay:5,quick:20,refer:18,releas:19,repack_stream:10,report:[11,15],reshap:[5,6],setup:20,solut:16,start:20,statu:[17,21],submodul:[4,5,6,7,8,9,10,11,12,13,14],subpackag:[3,4,8,9],talk:18,templat:12,tf_to_hl:4,top:0,trace:1,transpose_opt:10,tutori:17,uninstal:20,util:[4,13],vivado_report:11,vivado_templ:12,vivado_writ:14,welcom:17,work:16,write:1,writer:14,yaml:0}})