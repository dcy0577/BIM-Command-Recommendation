import sys
import json
import logging
import pathlib
import cloudpickle
import torch
import triton_python_backend_utils as pb_utils
# make sure we are using the customized transformers4rec
sys.path.insert(0, '/transformers4rec')

from merlin.systems.triton.utils import triton_error_handling, triton_multi_request
from merlin.systems.triton import  _convert_tensor
from merlin.systems.triton.export import _convert_string2pytorch_dtype

# Set up logging, the pb_utils.Logger cant log json 
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _parse_io_config(io_config):
    io = []
    for conf in io_config:
        io.append({"name": conf["name"]})
    return io

def _get_device_name(kind, device_id):
    if kind == "GPU":
        return "cuda:" + device_id
    if kind == "CPU":
        return "cpu"
    # unspecified device
    return ""

def _get_device(kind, device_id, model):
    device_name = _get_device_name(kind, device_id)
    if device_name == "":
        for param in model.parameters():
            return param.device
        raise pb_utils.TritonModelException("Cannot determine model device")
    return torch.device(device_name)


class TritonPythonModel:
    """The Python back-end for PyTorch serving"""

    def initialize(self, args):

        logger.info("Initializing model instance ")

        self._model_config = json.loads(args["model_config"])
        logger.info(f"Model config: {self._model_config}")
        self._inputs = _parse_io_config(self._model_config["input"])
        self._outputs = _parse_io_config(self._model_config["output"])

        self._kind = args["model_instance_kind"]
        self._device_id = args["model_instance_device_id"]

        self._infer_mode = torch.inference_mode(mode=True)
        self._infer_mode.__enter__()

        # Arg parsing
        repository_path = pathlib.Path(args["model_repository"])
        model_version = str(args["model_version"])

        # Handle bug in Tritonserver 22.06
        # model_repository argument became path to model.py
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent

        model_path = repository_path / model_version / "model.pkl"

        # Load the pickled PyTorch model
        with open(str(model_path), "rb") as model_file:
            self.model = cloudpickle.load(model_file)
        
        logger.info("pkl model loaded!!!")  
        
        self._device = _get_device(self._kind, self._device_id, self.model)

        logger.info(f"Device: {self._device}")

        # Load the state dict of the PyTorch model
        # model_path = repository_path / model_version / "model.pth"
        # self.model.load_state_dict(torch.load(str(model_path)))
        self.model.to(self._device)
        self.model.eval()
        
        logger.info("Model loaded!!!")   

    @triton_multi_request
    @triton_error_handling
    def execute(self, request):
        """Predicts the input batches by running through a PyTorch predict function."""

        logger.info("Executing model instance started!!!")

        # Convert the input data to dict to pass it into the PyTorch model
        input_dict = {}
        for io in self._inputs:
            tensor = pb_utils.get_input_tensor_by_name(
                request, io["name"]
            ).to_dlpack()
            tensor = torch.from_dlpack(tensor).to("cuda")
            input_dict[io["name"]] = tensor
        
        logger.info("Inference started, input_dict: ")
        logger.info(input_dict)
            
        # Call forward function to get the predictions
        output = self.model(input_dict) # should be a tuple

        logger.info("Inference down, original output from model: ")
        logger.info(output)

        # Convert the output to the Triton format
        output_tensors = []
        for i in range(len(self._outputs)):
            io = self._outputs[i]
            tensor = output[i].detach()
            tensor = pb_utils.Tensor.from_dlpack(io["name"], tensor)
            output_tensors.append(tensor)

        inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)

        return inference_response


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        logger.info("Removing model instance")
        self._infer_mode.__exit__(exc_type=None, exc_value=None, traceback=None)
