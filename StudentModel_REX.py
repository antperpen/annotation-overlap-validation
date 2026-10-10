## Student model based on ResNeXt-101 32x8d (weights from MicheleDamian/prostate-gleason-dataset)
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

import os

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

MODEL_WEIGHTS = 'models/resnext-101-32x8d.pth'
# TARGET_MPP is the resolution of the training images (0.452-0.5032 um/px in the original
# dataset, 0.48 um/px in the Radboud ones). It is used to choose the read layer of each WSI and to
# adjust the patch size and the scanning stride (see resolve_extraction in run_PROMETEO.py)
TARGET_MPP = 0.48
PATCH_SIZE = 512
OVERLAP = 256
STRIDE = PATCH_SIZE - OVERLAP
INPUT_SIZE = 256
RESAMPLE = Image.NEAREST
# These weights were trained with raw pixels in [0,1] (ToTensor only). Applying the ImageNet
# statistics shifts the input and collapses the output into a single class (verified: 25%
# accuracy with ImageNet versus 85% without normalization on the test set of the authors'
# dataset), so NORMALIZE must remain None.
NORMALIZE = None
N_CLASSES = 4


class StudentModel(nn.Module):
    def __init__(self, checkpoint_path=None, device='cpu', use_embedding=False):
        super(StudentModel, self).__init__()

        # 1. Recreate the EXACT base architecture as indicated by the repository
        base_model = models.resnext101_32x8d(pretrained=False)
        # The repository maps the 2048 features directly to 4 classes
        base_model.fc = nn.Linear(base_model.fc.in_features, N_CLASSES)

        # 2. Load the downloaded weights BEFORE modifying the structure of the network
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            base_model.load_state_dict(state_dict)
            print(f"MicheleDamian weights successfully loaded.")

        # 3. Separate the convolutional layers (Feature Extractor) from the final layer
        self.conv_layers = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_feat_in = base_model.fc.in_features  # 2048

        self.use_embedding = use_embedding
        self.N_CLASSES = N_CLASSES

        # 4. Define the head
        if self.use_embedding:
            # Option A: if the 128 embedding is required
            # (e.g. for Knowledge Distillation).
            # NOTE: These layers are initialized from scratch.
            self.E = 128
            self.embedding = nn.Linear(self.fc_feat_in, self.E)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(self.E, self.N_CLASSES)
        else:
            # Option B (Recommended): Use the model as it comes in the repository.
            # The FC layer that ALREADY has the trained weights loaded is reused.
            self.fc = base_model.fc

    def forward(self, x):
        # Extract features (Output: [Batch, 2048, 1, 1])
        x = self.conv_layers(x)

        # Flatten the spatial tensor into a 1D vector: [Batch, 2048]
        x = torch.flatten(x, 1)

        if self.use_embedding:
            emb = self.embedding(x)
            emb_relu = self.relu(emb)
            out = self.fc(emb_relu)
            return out # (or return out, emb_relu if it is needed for the loss)
        else:
            out = self.fc(x)
            return out


def build_model(device):
    """
    Builds the ResNeXt-101 32x8d model with the weights downloaded from the MicheleDamian
    repository, sets it in evaluation mode and on the given device.
    """
    model = StudentModel(checkpoint_path=MODEL_WEIGHTS, device=device)
    model = model.to(device)
    model.eval()
    return model


def predict(model, patch):
    """
    Returns (logits, probabilities) for the batch 'patch', which is the same output structure
    as the one offered by the densenet121 student model. ResNeXt only returns logits, so the
    softmax is applied here to complete the tuple.
    """
    logits = model(patch)
    classes = torch.nn.functional.softmax(logits, dim=1)
    return logits, classes
