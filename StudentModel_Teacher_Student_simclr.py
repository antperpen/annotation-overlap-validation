## Student model based on densenet121 with the backbone pretrained with SimCLR
## (weights student_model_Teacher_Student_simclr_exp7.pt, Teacher_Student pipeline from Examode)
##
## This file is the model module imported by run_PROMETEO.py. Every model file that can be
## imported by run_PROMETEO.py must export the same common interface:
##   - StudentModel : model class
##   - build_model(device) : returns the loaded instance, in evaluation mode and on the device
##   - predict(model, patch) : returns the tuple (logits, probabilities) for a patch
##   - INPUT_SIZE  : input size expected by the network
##   - TARGET_MPP  : micrometres per pixel of the images used to train the model
##   - RESAMPLE    : filter used to rescale the patch to INPUT_SIZE (PIL.Image constant)
##   - NORMALIZE   : (mean, std) used to normalize the input, or None if the weights were
##                   trained with raw pixels in [0,1]
##   - PATCH_SIZE  : size of the patch extracted from the WSI
##   - STRIDE      : scanning stride used to extract the patches
##   - N_CLASSES   : number of output Gleason grades
##   - MODEL_WEIGHTS : path of the model weights
##
## The architecture is identical to the one in StudentModel_densenet.py (same densenet121 backbone
## and same embedding head); the difference lies in the pretraining of the backbone (self-supervised
## SimCLR instead of ImageNet/from scratch, see Student_Inference_images.py in the Teacher_Student
## repository) and in the loaded fine-tuning weights. It has been verified, by loading the
## checkpoint, that the shape of its layers (embedding 1024->128, fc 128->4) matches this class.
##
## As in StudentModel_densenet, the model was saved complete with torch.save, so its inference
## is done by calling the model with forward(x, None).

import sys

import torch
from PIL import Image
# These imports are not used directly in the file, but they are needed to deserialize the
# weights, since the checkpoint stores references to those modules.
from torch.utils import data
import torch.nn.functional as F
import torch.utils.data

MODEL_WEIGHTS = 'models/student_model_Teacher_Student_simclr_exp7.pt'
INPUT_SIZE = 224
# These weights were trained with the standard ImageNet normalization, like
# StudentModel_densenet.py
NORMALIZE = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# Same resolution and extraction geometry as StudentModel_densenet.py: 750 px at 0.25 um/px
# (highest resolution layer of VALME) are 187.5 um of tissue rescaled to 224x224
TARGET_MPP = 0.25
PATCH_SIZE = 750
OVERLAP = 0
STRIDE = PATCH_SIZE - OVERLAP
RESAMPLE = Image.BILINEAR
N_CLASSES = 4


class StudentModel(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(StudentModel, self).__init__()

        pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=imageNet_weights)
        fc_input_features = pre_trained_network.classifier.in_features

        self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])

        if (torch.cuda.device_count()>1):
            self.conv_layers = torch.nn.DataParallel(self.conv_layers)

        self.fc_feat_in = fc_input_features
        self.N_CLASSES = 4

        self.E = 128

        self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
        self.fc = torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES)
        self.relu = torch.nn.ReLU()

    def forward(self, x, conv_layers_out):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            m_multiclass = torch.nn.Softmax()

            if x is not None:
                conv_layers_out=self.conv_layers(x)

                n = torch.nn.AdaptiveAvgPool2d((1,1))
                conv_layers_out = n(conv_layers_out)

                conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

            embedding_layer = self.embedding(conv_layers_out)
            embedding_layer = self.relu(embedding_layer)
            embedding_layer = torch.nn.functional.dropout(embedding_layer, p=0.2, training=self.training)

            logits = self.fc(embedding_layer)

            output_fcn = m_multiclass(logits)

            return logits, output_fcn


def build_model(device):
    """
    Loads the complete densenet121 model (SimCLR backbone) serialized in MODEL_WEIGHTS, sets it
    in evaluation mode and on the given device.

    The weights were saved with torch.save from a script where StudentModel lived in
    __main__, so the class must be registered in that module to be able to deserialize them
    when this file is imported as a module.
    """
    sys.modules['__main__'].StudentModel = StudentModel
    model = torch.load(MODEL_WEIGHTS, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def predict(model, patch):
    """
    Returns the tuple (logits, probabilities) already computed by the student model itself.
    """
    logits, classes = model(patch, None)
    return logits, classes
