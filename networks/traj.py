import torch
import torch.nn as nn
import einops
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len= 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class PosVAE(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, latent_dim, traj_len=5, pos_dim=10, conditional=False, condition_dim=None):

        super().__init__()

        self.latent_dim = latent_dim
        self.conditional = conditional

        if self.conditional and condition_dim is not None:
            # input_dim = in_dim + condition_dim
            input_dim = in_dim
            dec_dim = latent_dim + condition_dim
        else:
            input_dim = in_dim
            dec_dim = latent_dim
        self.enc_MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU())
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)
        self.pos_enc = PositionalEncoding(1, max_len=in_dim+1)
        self.pos_MLP = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ELU())
        self.dec_MLP = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, in_dim))

    def forward(self, x, c=None, return_pred=False):
        B = x.shape[0]
        # import ipdb; ipdb.set_trace()
        x = x.reshape(B, -1)
        pos_x = self.pos_enc(x.transpose(0, 1).unsqueeze(-1))
        pos_x = self.pos_MLP(pos_x.transpose(0, 1)[:, :, 0])
        if self.conditional and c is not None:
            # inp = torch.cat((pos_x, c), dim=-1)
            inp = pos_x
        else:
            inp = pos_x
        h = self.enc_MLP(inp)
        mean = self.linear_means(h)
        log_var = self.linear_log_var(h)
        z = self.reparameterize(mean, log_var)*0
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)
        recon_loss, KLD = self.loss_fn(recon_x, x, mean, log_var)
        if not return_pred:
            return recon_loss, KLD
        else:
            return recon_x, recon_loss, KLD

    def loss_fn(self, recon_x, x, mean, log_var):
        recon_loss = torch.sum((recon_x - x) ** 2, dim=1)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        return recon_loss, KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)
        return recon_x


class TrajAffCVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                z_scale=2.0, traj_len=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_scale = z_scale
        self.cvae = PosVAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim, traj_len=traj_len)

    def forward(self, context, target_hand, contact_point=None, return_pred=False):
        batch_size = context.shape[0]
        condition_context = context
        pred_hand, recon_loss, KLD = self.cvae(target_hand, c=condition_context, return_pred=return_pred)
        pred_hand = pred_hand.reshape(batch_size, -1, 2)
        KLD = KLD.sum()
        traj_loss = recon_loss.mean()
        return pred_hand, traj_loss, KLD

    def inference(self, context, contact_point=None):
        condition_context = context
        z = self.z_scale * torch.randn([context.shape[0], self.latent_dim], device=context.device)*0
        recon_x = self.cvae.inference(z, c=condition_context)
        return recon_x
    
    
