"""Run a trained segmentation model over a list of image tiles."""

import numpy as np
import torch


def run_inference_on_tiles(model, tiles, device, batch_size=8):
    """Run the model over a list of tiles and return one predicted class map per tile.

    Tiles are grouped into batches purely for speed. Each tile still receives its own
    independent prediction, since convolutions do not mix information across batch items.
    """
    predictions = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch = tiles[start:start + batch_size]
            batch_tensor = torch.from_numpy(np.stack(batch)).to(device)

            logits = model(batch_tensor)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)

    return predictions
