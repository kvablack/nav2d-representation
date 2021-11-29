import torch
import numpy as np
from matplotlib import cm


def render(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = (obs + 1) / 2
    img = np.zeros_like(obs)
    img[0][obs[0] != 0] = 1
    img[1][obs[1] != 0] = 1
    img[2][obs[2] != 0] = 1
    return img


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def pairwise(a, b):
    assert a.shape == b.shape
    batch_size = a.shape[0]
    a_expand = torch.broadcast_to(a, [batch_size] + [-1] * len(a.shape))
    b_expand = torch.broadcast_to(b, [batch_size] + [-1] * len(b.shape))
    a_flat = a_expand.reshape([batch_size ** 2] + list(a.shape[1:]))
    b_flat = b_expand.transpose(0, 1).reshape([batch_size ** 2] + list(b.shape[1:]))
    return a_flat, b_flat


def pairwise_l1_distance(x):
    a, b = pairwise(x, x)
    return torch.linalg.norm(a - b, ord=1, dim=-1)


def angular_distance(a, b):
    assert a.shape == b.shape
    assert len(a.shape) == 2
    numerator = torch.sum(a * b, dim=-1)
    denominator = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    cos_similarity = numerator / denominator
    return torch.atan2(torch.sqrt(torch.clamp(1 - cos_similarity ** 2, min=1e-9)), cos_similarity)


def mico_distance(a, b, beta):
    assert a.shape == b.shape
    assert len(a.shape) == 2
    norm = (torch.sum(a ** 2, dim=-1) + torch.sum(b ** 2, dim=-1)) / 2
    ang = angular_distance(a, b)
    return 500 * norm + ang, norm, ang


def grad_heatmap(obs, encoder):
    obs = torch.as_tensor(obs, device="cuda").requires_grad_()
    obs.grad = None
    norm = torch.linalg.vector_norm(encoder(obs.unsqueeze(0)).squeeze(0), ord=1)
    norm.backward()

    img = render(obs)
    img = img.repeat(2, axis=1).repeat(2, axis=2)

    heatmap = obs.grad.cpu().numpy()
    # grad_heatmap = np.sum(grad_heatmap, axis=0)
    heatmap /= np.max(heatmap, axis=(1, 2), keepdims=True)
    obstacle_heatmap = cm.gray(heatmap[0])[:, :, :3] \
        .transpose([2, 0, 1]) \
        .repeat(2, axis=1).repeat(2, axis=2)
    agent_heatmap = cm.gray(heatmap[1])[:, :, :3] \
        .transpose([2, 0, 1]) \
        .repeat(2, axis=1).repeat(2, axis=2)

    obstacle_heatmap[:, :, 0] = [[0], [1], [0]]
    agent_heatmap[:, :, -1] = [[0], [1], [0]]
    agent_heatmap[:, :, 0] = [[0], [1], [0]]
    img[:, :, -1] = [[0], [1], [0]]

    return img, agent_heatmap, obstacle_heatmap


def perturb_heatmap(obs, encoder):
    c, h, w = obs.shape
    encoded = encoder(
        torch.as_tensor(obs, device="cuda").unsqueeze(0)
    )\
        .squeeze(0).detach().cpu().numpy()

    obs_perturbed = np.broadcast_to(obs, [h * w, 3, h, w]).copy()
    mask = (-np.eye(h * w) * 2 + 1).reshape(h * w, h, w)
    obs_perturbed[:, 0] *= mask
    encoded_perturbed = encoder(
        torch.as_tensor(obs_perturbed, device="cuda")
    )\
        .detach().cpu().numpy()

    distances = np.linalg.norm(encoded - encoded_perturbed, ord=1, axis=-1).reshape(h, w)
    player_pos = np.argwhere(obs[1] == 1)[0]
    distances[player_pos[0], player_pos[1]] = 0
    goal_pos = np.argwhere(obs[2] == 1)
    if len(goal_pos) > 0:
        distances[goal_pos[0, 0], goal_pos[0, 1]] = 0
    distances /= np.max(distances)
    heatmap = cm.gray(distances)[:, :, :3] \
        .transpose([2, 0, 1]) \
        .repeat(2, axis=1).repeat(2, axis=2)
    # norm_distances = np.abs(
    #     np.sum(encoded**2, axis=-1) - np.sum(encoded_perturbed**2, axis=-1)
    # ).reshape(h, w)
    # norm_distances /= np.max(norm_distances)
    # ang_distances = np.arccos(np.clip(
    #     (encoded * encoded_perturbed).sum(axis=-1)
    #     / (np.linalg.norm(encoded, axis=-1) * np.linalg.norm(encoded_perturbed, axis=-1)),
    #     -1,
    #     1,
    # )).reshape(h, w)
    # ang_distances /= np.pi
    # norm_heatmap = cm.gray(norm_distances)[:, :, :3] \
    #     .transpose([2, 0, 1]) \
    #     .repeat(2, axis=1).repeat(2, axis=2)
    # ang_heatmap = cm.gray(ang_distances)[:, :, :3] \
    #     .transpose([2, 0, 1]) \
    #     .repeat(2, axis=1).repeat(2, axis=2)

    img = render(obs)
    img = img.repeat(2, axis=1).repeat(2, axis=2)
    img[:, :, -1] = [[0], [1], [0]]
    heatmap[:, :, 0] = [[0], [1], [0]]
    # norm_heatmap[:, :, -1] = [[0], [1], [0]]
    # ang_heatmap[:, :, 0] = [[0], [1], [0]]

    return img, heatmap


def perturb_heatmap_push(obs, encoder, env):
    heatmap = np.empty([obs.shape[1], obs.shape[2]], dtype=np.float32)
    player_pos = np.argwhere(obs[1] == 1)[0]
    player_block_ind = np.array(np.broadcast_arrays(*env._block_ind(player_pos))).reshape(2, -1).T
    obs_encoded = encoder(
        torch.as_tensor(obs, device="cuda").unsqueeze(0)
    ).squeeze(0).detach().cpu().numpy()
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            block_ind = env._block_ind((y, x))
            if (np.array([y, x]) == player_block_ind).all(axis=1).any():
                heatmap[y, x] = 0
                continue
            obs_perturb = obs.copy()
            obs_perturb[0] = -1
            obs_perturb[0][block_ind] = 1
            obs_perturb_encoded = encoder(
                torch.as_tensor(obs_perturb, device="cuda").unsqueeze(0)
            ).squeeze(0).detach().cpu().numpy()
            heatmap[y, x] = np.linalg.norm(obs_encoded - obs_perturb_encoded, ord=1)

    heatmap /= np.max(heatmap)
    heatmap[tuple(player_pos)] = 1
    heatmap = cm.gray(heatmap)[:, :, :3] \
        .repeat(2, axis=0).repeat(2, axis=1) \
        .transpose([2, 0, 1])

    img = render(obs)
    img = img.repeat(2, axis=1).repeat(2, axis=2)
    img[:, :, -1] = [[0], [1], [0]]
    heatmap[:, :, 0] = [[0], [1], [0]]

    return img, heatmap