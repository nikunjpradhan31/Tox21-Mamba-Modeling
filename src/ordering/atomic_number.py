import torch


def get_order(data, descending=False, **kwargs):
    """
    Returns a permutation of nodes sorted by atomic number within each graph in the batch.
    """
    device = data.batch.device
    num_nodes = data.num_nodes

    # Extract atomic numbers (z)
    if hasattr(data, "z") and data.z is not None:
        z = data.z.to(torch.float64)
    else:
        z = data.x[:, 0].to(torch.float64)

    # Add a small amount of random noise to break ties consistently
    noise = torch.rand(num_nodes, device=device, dtype=torch.float64) * 0.1
    scores = z + noise

    if descending:
        scores = -scores

    # Normalize scores to be strictly between 0 and 1
    scores = scores - scores.min()
    max_score = scores.max()
    if max_score > 0:
        scores = scores / (max_score + 1.0)

    # Add batch index to group nodes by graph
    batch_scores = data.batch.to(torch.float64) + scores

    perm = torch.argsort(batch_scores)
    return perm
