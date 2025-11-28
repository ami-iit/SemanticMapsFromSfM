import torch
from torchvision import transforms
from PIL import Image
import os
from glob import glob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True # Needed because of a Pytorch 1.9 known issue. 
    model = torch.hub.load(
        "pytorch/vision:v0.10.0",
        "deeplabv3_resnet50",
        pretrained=True
    )
    model.eval().to(DEVICE)
    return model


def main(images_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_paths = sorted(glob(os.path.join(images_dir, "*.jpg")))
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)["out"]  # [1, num_classes, H, W]
        pred = out.argmax(1).squeeze(0).cpu().numpy().astype("uint8")

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, base + ".png")
        Image.fromarray(pred).save(out_path)
        print("Saved", out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Data/images") # After debugging, replace default with required=True
    parser.add_argument("--out_dir", default="/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Data/semantic_masks")
    args = parser.parse_args()
    main(args.images_dir, args.out_dir)