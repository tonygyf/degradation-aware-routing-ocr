"""InternVL model-card preprocessing, pinned official candidate (Apache-2.0).

Reference: OpenGVLab/InternVL3_5-8B at
9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79, README inference example.
The IJDAR author's exact preprocessing configuration remains unavailable.
"""


def choose_grid(width: int, height: int, max_tiles: int = 12, tile_size: int = 448):
    if any(
        not isinstance(value, int) or value <= 0 for value in (width, height, max_tiles, tile_size)
    ):
        raise ValueError("Image dimensions, tile budget and tile size must be positive integers")
    ratios = sorted(
        {
            (i, j)
            for n in range(1, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_tiles
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    best, distance = (1, 1), float("inf")
    for columns, rows in ratios:
        candidate_distance = abs(width / height - columns / rows)
        if candidate_distance < distance or (
            candidate_distance == distance and width * height > 0.5 * tile_size**2 * columns * rows
        ):
            best, distance = (columns, rows), candidate_distance
    return best


def image_tiles(image, max_tiles=12, tile_size=448):
    image = image.convert("RGB")
    columns, rows = choose_grid(*image.size, max_tiles, tile_size)
    resized = image.resize((columns * tile_size, rows * tile_size))
    tiles = [
        resized.crop((x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size))
        for y in range(rows)
        for x in range(columns)
    ]
    if len(tiles) > 1:
        tiles.append(image.resize((tile_size, tile_size)))
    return tiles


def load_pixels(path, max_tiles=12, tile_size=448):
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    transform = transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with Image.open(path) as image:
        tiles = image_tiles(image, max_tiles, tile_size)
    return torch.stack([transform(tile) for tile in tiles])
