import qrcode
from math import ceil

from PIL import ImageOps
import numpy as np
import torch

from custom_nodes.DTAIComfyVariables import variables


class QrCodeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "link": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "https://doubtech.ai",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "create_qr_code"

    # OUTPUT_NODE = False

    CATEGORY = "DoubTech/Loaders"

    def create_qr_code(self, link):
        # Data to encode
        data = variables.apply(link)

        # Desired size in pixels
        size = 768

        # Size of the border in blocks
        border = 5

        # Version of the QR code, could vary depending on the length of data
        version = 1

        # Number of modules (blocks) based on the version
        modules = version * 4 + 17

        # Calculate box size so that (box_size * modules + 2 * border * box_size) is close to desired size
        box_size = ceil(size / (modules + 2 * border))

        qr = qrcode.QRCode(
            version=version,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=box_size,
            border=border,
        )

        qr.add_data(data)
        qr.make(fit=True)

        # Generate QR Code
        i = qr.make_image(fill='black', back_color='white')
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)
        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "QRCode": QrCodeNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "QRCode": "QR Code"
}
