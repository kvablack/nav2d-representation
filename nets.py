import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        c, h, w = obs_dim
        
        self.grid = (
            torch.stack(
                torch.meshgrid(torch.arange(0, h) / (h - 1), torch.arange(0, w) / (w - 1)), dim=0
            ) * 2 - 1
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
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
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
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
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
        self.fc_mu = torch.nn.Linear(self.encoder.output_dim, latent_dim)
        self.fc_log_var = torch.nn.Linear(self.encoder.output_dim, latent_dim)
        self.output_dim = latent_dim

    def forward(self, x):
        return self.fc_mu(self.encoder(x))

    def mu_log_var(self, x):
        encoded = self.encoder(x)
        return self.fc_mu(encoded), self.fc_log_var(encoded)


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
