import requests
import jax.numpy as np
from io import BytesIO
from PIL import Image
from einops import reduce

# typing
from jax import Array


def load_image(url: str, target_size: int) -> Array:
    r = requests.get(url)
    img = Image.open(BytesIO(r.content))
    img = img.resize((target_size, target_size))
    img = np.float32(img) / 255.0
    img = img.at[..., :3].set(img[..., :3] * img[..., 3:])
    img = reduce(img, 'h w c -> h w', 'mean')
    return img


def load_emoji(emoji: str, target_size: int) -> Array:
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
    return load_image(url, target_size)
