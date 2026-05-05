from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, Lambda
from Dataset.ImageDataset_visual import VisualDataset
from Dataset.ImageDataset_ex import exVisualDataset
from torch.utils.data import DataLoader
from Dataset.ImageDataset_infer import InfernanoDataset
from utils.utils import mask_to_tensor, expand_channels, CFG



# Image preprocessing pipeline
transform = Compose([
    Resize((CFG.img_size)),
    ToTensor(),
    expand_channels,
])

# Mask preprocessing pipeline
mask_transform = Compose([
    Resize((CFG.img_size), interpolation=InterpolationMode.NEAREST),
    Lambda(mask_to_tensor)
])

def single_collate(batch):
    return batch[0]

# Streaming loader (PNG image + PNG mask)
# - Used for sequential inference / visualization
def prepare_streaming_loader():
    dataset = VisualDataset(
        root_path='./Data/RealCAC_dataset/image',
        label_path='./Data/RealCAC_dataset/2d_label.csv',
        mask_path='./Data/RealCAC_dataset/label',
        transform=transform,
        mask_transform=mask_transform,
        mode='train'
    )

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Extended loader (PNG image + NumPy mask)
# - Used for external datasets (e.g., Mindray)
def prepare_ex_loader():
    dataset = exVisualDataset(
        root_path='./Data/external_dataset/image',
        label_path='./Data/external_dataset/Mindrayfinal_labels.csv',
        mask_path='./Data/external_dataset/mask',
        transform=transform,
        mask_transform=mask_transform,
        mode='train'
    )

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


# Inference-only loader (image only, no mask)
# - Used for real-time / deployment pipeline
def nano_streaming_loader():
    out_dir = "./Data/intrCPR_dataset/image/003_3_6"

    dataset = InfernanoDataset(
        root_path=out_dir,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=single_collate,
        pin_memory=False
    )