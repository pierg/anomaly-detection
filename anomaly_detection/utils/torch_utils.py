"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
from pathlib import Path
from torchviz import make_dot
from torchinfo import summary
from torchview import draw_graph

def save_model_info(model: nn.Module, input_tensor: torch.Tensor, folder: Path, id="") -> None:
    """
    Save model architecture views, info, and ONNX export to files, with model state management.
    Also, clean up any intermediate files generated in the process.

    Args:
        model (nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): A tensor representative of a single input to the model.
        folder (Path): The directory path where the files will be saved.
    """

    # Ensure the required folders exist
    required_folders = ['onnx', 'torchviz', 'torchinfo', 'torchview']
    for subfolder in required_folders:
        (folder / subfolder).mkdir(parents=True, exist_ok=True)

    # Check model's current mode and set to eval if necessary
    was_training = model.training
    model.eval()

    name = model.__class__.__name__

    try:
        # Export the model to ONNX format
        onnx_path = folder / 'onnx' / f'{name}_{id}.onnx'

        torch.onnx.export(model,                                  # model being run
                    input_tensor,                                 # model input (or a tuple for multiple inputs)
                    onnx_path,                                    # where to save the 
                    verbose=False,                                # print out a verbose ONNX representation of the model
                    export_params=True,                           # store the trained parameter weights inside the model file
                    opset_version=11,                             # the ONNX version to export the model to
                    do_constant_folding=True,                     # whether to execute constant folding for optimization
                    input_names = ['input'],                      # the model's input names
                    output_names = ['output'],                    # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},     # variable length axes
                                    'output': {0: 'batch_size'}})
        print(f"Model exported to ONNX format at {onnx_path}")

        # Save the graphical representation of the model
        try:
            output = model(input_tensor)
            parameters = dict(model.named_parameters())
            graph = make_dot(output, params=parameters, show_attrs=False, show_saved=True)
            torchviz_path = folder / 'torchviz' / f'{name}_{id}'
            graph.render(torchviz_path, format="pdf", cleanup=True)
            print(f"Torchviz graph saved to {torchviz_path}.pdf")
        except Exception as e:
            print(f"Failed to generate torchviz graph: {e}")

        # Save the summary of the model
        summary_path = folder / 'torchinfo' / f'{name}_{id}.txt'
        model_summary = summary(model, input_data=input_tensor, verbose=0)
        with summary_path.open("w") as f:
            f.write(str(model_summary))
        print(f"Model summary saved to {summary_path}")

        # Generate and save the torchview graph
        model_graph = draw_graph(model, input_data=input_tensor, expand_nested=True,
                                 hide_inner_tensors=True, hide_module_functions=False,
                                 roll=False, depth=20)
        torchview_path = folder / 'torchview' / f'{name}_{id}'
        model_graph.visual_graph.render(torchview_path, format='pdf', cleanup=True)
        print(f"Torchview graph saved to {torchview_path}.pdf")

    finally:
        # Ensure the model is returned to its original training state
        if was_training:
            model.train()

