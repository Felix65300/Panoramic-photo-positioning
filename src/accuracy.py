import os
import sys

from torchvision.models import resnet18

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_dir = os.path.join(parent_dir, 'Custom_model')
resnet18_dir = os.path.join(parent_dir, "resnet18")
Project_Root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(Project_Root)
from src