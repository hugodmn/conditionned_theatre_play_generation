import torch 
import onnx 
import onnxruntime as ort 
from optimum.onnxruntime import ORTModelForAudioClassification
from onnxruntime import SessionOptions
#from onnx_tf.backend import prepare
import numpy as np 
from model_V2 import LLM, Config
import torch
import json

def loading_model(model, weights_path):
   
        #Load json file in a dict
        config = Config(
        vocab_size = 1070,
        emb_size = 384,
        head_nb = 6,
        block_nb = 6,
        block_size = 256,
        dropout=0.0,
        )

        inference = model(config)
        inference.load_state_dict(torch.load(weights_path))
        inference.eval()

        return inference


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    print("Starting quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8,  
                     use_external_data_format=False)

    print(f"Quantized model saved to: {quantized_model_path}")


def saving_onnx(torch_model, saved_name, quantization = False, export = True):
    


    #torch_in = torch.randn( 1, torch_model.config.block_size, requires_grad=True, dtype=torch.int32)
    torch_in = torch.randint(0, 1070, (1, 256), dtype=torch.int32)
    print(torch_in)
    print(torch_in.shape)
    torch_out = torch_model(torch_in)
    # dynamic_axes = {
    #     'input' : {1: 'audio_len'},
    #  }
    # Export the model
    if export :
        torch.onnx.export(torch_model,               # model being run
                        torch_in,                         # model input (or a tuple for multiple inputs)
                        saved_name+'.onnx',   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        
                        opset_version=14,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'},
                                 'output': {0: 'batch_size', 1: 'seq_len'}} )  # variable length axes

        if quantization :
            quantize_onnx_model(saved_name+".onnx", saved_name+".quant.onnx")

    return torch_in, torch_out

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_test(onnx_path, torch_in, torch_out):

    sess_options = SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": to_numpy(torch_in)})[0]

    np.testing.assert_allclose(to_numpy(torch_out)[0], onnx_out[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# def export_to_tf(onnx_path, tf_path):
#     onnx_model = onnx.load(onnx_path)
#     tf_rep = prepare(onnx_model)
#     tf_rep.export_graph(tf_path)
    

if __name__ == "__main__":
    weights_file_name = "char_level"
    quantization = True

    model_inf = loading_model(LLM,'checkpoints/finetuned/bpe/model.pt' )
    torch_in, torch_out = saving_onnx(model_inf, 'ONNX_saved/bpe/exported_model', quantization, export = True)
    onnx_test(onnx_path='ONNX_saved/bpe/exported_model.onnx',torch_in = torch_in, torch_out = torch_out)
