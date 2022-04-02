from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        c, h, w = obs_dim

        self.grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, h) / (h - 1), torch.arange(0, w) / (w - 1)
                ),
                dim=0,
            )
            * 2
            - 1
        ).cuda()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=c + 2,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d((2, 2)),
            # torch.nn.Conv2d(
            #     in_channels=96,
            #     out_channels=128,
            #     kernel_size=3,
            #     stride=1,
            #     padding="same",
            # ),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d((2, 2)),
            # torch.nn.Conv2d(
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=3,
            #     stride=1,
            #     padding="same",
            # ),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d((5, 5)),
        )

        self.output_dim = self.conv(torch.zeros([1, c + 2, h, w])).flatten().shape[0]

        print("Embed dim:", self.output_dim)

    def forward(self, obs):
        # return obs.flatten(start_dim=1).float()
        # ret = torch.argmax(obs[:, 1].flatten(start_dim=1), dim=-1)
        # return ret.unsqueeze(-1).float()
        grid_expand = self.grid.expand(obs.shape[0], -1, -1, -1)
        combined = torch.cat([obs, grid_expand], dim=1)
        return self.conv(combined).flatten(start_dim=1)


class DeterministicEncoder(torch.nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(obs_dim)
        self.fc = torch.nn.Linear(self.encoder.output_dim, latent_dim)
        self.output_dim = latent_dim

    def forward(self, x):
        return self.fc(self.encoder(x))


class StochasticEncoder(torch.nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(obs_dim)
        assert self.encoder.output_dim % 2 == 0
        # self.fc_mu = torch.nn.Linear(self.encoder.output_dim, latent_dim)
        # self.fc_log_var = torch.nn.Linear(self.encoder.output_dim, latent_dim)
        self.output_dim = self.encoder.output_dim // 2

    def forward(self, x):
        return self.encoder(x)[:, : self.output_dim]
        # return self.fc_mu(self.encoder(x))

    def mu_log_var(self, x):
        encoded = self.encoder(x)
        return encoded[:, : self.output_dim], encoded[:, self.output_dim :]


class InverseDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, embed, embed_next):
        return self.fc(torch.cat([embed, embed_next], dim=-1))


class StochasticForwardDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.fc_mu = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + action_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, embed_dim),
        )
        self.fc_log_var = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + action_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, embed_dim),
        )

    def forward(self, embed, action):
        x = torch.cat([embed, F.one_hot(action, num_classes=self.action_dim)], dim=-1)
        return self.fc_mu(x), self.fc_log_var(x)


class ForwardDynamics(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embed_dim),
        )

    def forward(self, embed, action):
        x = torch.cat([embed, F.one_hot(action, num_classes=self.action_dim)], dim=-1)
        return self.fc(x)


class DQN(torch.nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, embed):
        return self.fc(embed)


class SideTuner(torch.nn.Module):
    def __init__(self, encoder, obs_dim):
        super().__init__()
        c, h, w = obs_dim
        self.encoder = encoder
        # self.side_encoder = deepcopy(encoder)
        self.side_encoder = Encoder((1, h, w))
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        # self.alpha = 0

    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        return a * self.encoder(x[:, :2]).detach() + (1 - a) * self.side_encoder(
            x[:, 2].unsqueeze(1)
        )
        # return torch.cat([self.encoder(x[:, :2]).detach(), self.side_encoder(x[:, 2].unsqueeze(1))], dim=1)


class Decoder(torch.nn.Module):
    def __init__(self, embed_dim, obs_dim):
        super().__init__()
        c, h, w = obs_dim

        self.grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, h) / (h - 1), torch.arange(0, w) / (w - 1)
                ),
                dim=0,
            )
            * 2
            - 1
        ).cuda()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=embed_dim + 2,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        grid_expand = self.grid.expand(x.shape[0], -1, -1, -1)
        x_expand = (
            x.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.grid.shape[2], self.grid.shape[2])
        )
        combined = torch.cat([x_expand, grid_expand], dim=1)
        return self.conv(combined)
