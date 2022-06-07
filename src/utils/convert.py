# from pathlib import Path
import torch
import onnx
import numpy as np
import onnxruntime
from argparse import ArgumentParser

from ..models.model import get_encoder, get_decoder, build_model
from ..config import Config
from .utils import seed_everything


enc_pth = "checkpoints/encoder.pth"
dec_pth = "checkpoints/decoder.pth"
enc_onnx_pth = "checkpoints/celeryLatexEncoder.onnx"
dec_onnx_pth = "checkpoints/celeryLatexDecoder.onnx"


def save_enc_dec_pth(model):
    torch.save(model.encoder.state_dict(), enc_pth)
    print(f"saved encoder to {enc_pth}")
    torch.save(model.decoder.state_dict(), dec_pth)
    print(f"saved decoder to {dec_pth}")


def convert_enc_dec(device="cpu"):
    encoder = get_encoder(
        in_chans=conf.in_chans,
        next_depths=conf.next_depths,
        next_dims=conf.next_dims,
        pdrop_path=conf.pdrop_path,
    ).to(device)

    decoder = get_decoder(
        model_dim=conf.mdim,
        num_head=conf.nhead,
        dropout=conf.pdrop,
        vocab_size=conf.vocab_size,
        max_seq_len=conf.max_seq,
        ff_dim = conf.ff_dim,
        dec_depth=conf.dec_depth,
    ).to(device)

    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(enc_pth, map_location=device))
    decoder.load_state_dict(torch.load(dec_pth, map_location=device))
    print("loaded encoder and decoder checkpoint")

    img = torch.randn(1, 1, 192, 896)
    tgt = torch.randint(8000, (1, 512))
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).bool()
    # out = model(img, tgt, True)
    memory = encoder(img)

    # Decoder
    dynamic_axes_dec={
        'memory' : {
            0 : 'batch_size',
            1: "seq_len",
        },
        "tgt": {
            0: "batch_size",
            1: "tgt_len",
        },
        "tgt_mask": {
            0: "batch_size",
            1: "tgt_len",
        },
        "predicted": {
            0: "batch_size",
            1: "seq_len",
        },
    }
    torch.onnx.export(
        decoder,
        (tgt, memory, tgt_mask),                         # model input (or a tuple for multiple inputs)
        dec_onnx_pth,   # where to save the model (can be a file or file-like object)
        input_names=['tgt', "memory", "tgt_mask"],   # the model's input names
        output_names=['predicted'], # the model's output names
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=14,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        dynamic_axes=dynamic_axes_dec,
        # verbose=True,
    )
    print(f"saved decoder onnx to {dec_onnx_pth}")

    # Encoder
    dynamic_axes_enc={
        'img' : {
            0 : 'batch_size',
            2: "height",
            3: "width"
        },
        'memory' : {
            0 : 'batch_size',
            1: "seq_len",
        },
    }
    torch.onnx.export(
        encoder,               # model being run
        (img, ),                         # model input (or a tuple for multiple inputs)
        enc_onnx_pth,   # where to save the model (can be a file or file-like object)
        input_names = ['img',],   # the model's input names
        output_names = ['memory'], # the model's output names
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=14,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        dynamic_axes=dynamic_axes_enc,
        # verbose=True,
    )
    print(f"saved decoder onnx to {enc_onnx_pth}")


def check_onnx():
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # onnx_model = onnx.load("checkpoints/celeryLatexOCR.onnx")
    # onnx.checker.check_model(onnx_model)
    # ort_session = onnxruntime.InferenceSession("checkpoints/celeryLatexOCR.onnx")


    # # compute ONNX Runtime output prediction
    # ort_inputs = {
    #     ort_session.get_inputs()[0].name: to_numpy(img),
    #     ort_session.get_inputs()[1].name: None,
    #     ort_session.get_inputs()[2].name: True,
    # }
    # ort_outs = ort_session.run(None, ort_inputs)
    # np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    ...


def main(conf: Config, device="cpu"):
    model = build_model(
        in_chans=conf.in_chans,
        model_dim=conf.mdim,
        num_head=conf.nhead,
        dropout=conf.pdrop,
        dec_depth=conf.dec_depth,
        ff_dim = conf.ff_dim,
        vocab_size=conf.vocab_size,
        max_seq_len=conf.max_seq,
        temperature=conf.temperature,
        next_depths=conf.next_depths,
        next_dims=conf.next_dims,
        pdrop_path=conf.pdrop_path,
        model_name=conf.model_name,
        device=device,
    )
    try:
        print(f"Loading checkpoint {conf.checkpoint}")
        model.load_state_dict(torch.load(conf.checkpoint, map_location=device))
    except Exception as e:
        print(f"Load checkpoint failed, {e}")
    model.eval()

    save_enc_dec_pth(model)
    convert_enc_dec(device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config", type=str, help=".json config file path")

    args = parser.parse_args([
        "-c", "src/config/config_convnext.json",
    ])
    conf = Config(args.config)
    seed_everything(conf.seed)
    main(conf=conf)
