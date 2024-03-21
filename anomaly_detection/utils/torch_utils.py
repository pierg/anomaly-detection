import torch
import torch.nn as nn
from pathlib import Path
from torchviz import make_dot
from torchinfo import summary
from torchview import draw_graph

def save_model_info(model: nn.Module, input_tensor: torch.Tensor, folder: Path) -> None:
    # Check if the folder exists, if not, create it
    folder.mkdir(parents=True, exist_ok=True)

    # Check model's current mode and set to eval if necessary
    was_training = model.training
    model.eval()

    try:
        # Export the model to ONNX format
        onnx_path = folder / 'onnx.onnx'
        torch.onnx.export(model, input_tensor, onnx_path, verbose=False, export_params=True, opset_version=11,
                          do_constant_folding=True, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f"Model exported to ONNX format at {onnx_path}")

        # Save the graphical representation of the model
        try:
            output = model(input_tensor)
            parameters = dict(model.named_parameters())
            graph = make_dot(output, params=parameters, show_attrs=False, show_saved=True)
            torchviz_path = folder / 'torchviz'
            graph.render(torchviz_path, format="pdf", cleanup=True)
            print(f"Torchviz graph saved to {torchviz_path}.pdf")
        except Exception as e:
            print(f"Failed to generate torchviz graph: {e}")

        # Save the summary of the model
        summary_path = folder / 'torchinfo.txt'
        model_summary = summary(model, input_data=input_tensor, verbose=0)
        with summary_path.open("w") as f:
            f.write(str(model_summary))
        print(f"Model summary saved to {summary_path}")

        # Generate and save the torchview graph
        try:
            model_graph = draw_graph(model, input_data=input_tensor, expand_nested=True,
                                     hide_inner_tensors=True, hide_module_functions=False,
                                     roll=False, depth=20)
            torchview_path = folder / 'torchview'
            model_graph.visual_graph.render(torchview_path, format='pdf', cleanup=True)
            print(f"Torchview graph saved to {torchview_path}.pdf")
        except Exception as e:
            print(f"Failed to generate torchview graph: {e}")
        
    finally:
        # Ensure the model is returned to its original training state
        if was_training:
            model.train()
