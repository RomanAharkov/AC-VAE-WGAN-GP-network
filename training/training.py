from models.models import Generator, Discriminator, Classifier, Encoder
from losses.loss_functions import disc_loss, gen_loss, class_loss, recon_loss, kl_loss
from data.dataset import HyperspectralDataset
import torch
from torch.utils.data import DataLoader
from utils.util_functions import get_next_batch, reparameterize


def train_models(num_epochs: int, class_num: int, latent_dim: int, channels: int, mode: str, learning_rate: float,
                 device: str, train_dataset: HyperspectralDataset, batch_size: int, num_workers: int, disc_cycles: int,
                 alpha: float, tau: int, recon_weight: float, gp_weight: float):

    scaler = torch.amp.GradScaler(device)

    # instantiate the models and the data
    gen = Generator(class_num, latent_dim, channels, mode=mode).to(device)
    disc = Discriminator(channels).to(device)
    if mode == "VAE":
        enc = Encoder(latent_dim, channels).to(device)
        enc_opt = torch.optim.Adam(enc.parameters(), lr=learning_rate)
    cls = Classifier(class_num).to(device)

    gen_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate)
    cls_opt = torch.optim.Adam(cls.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    train_iter = iter(train_loader)

    for epoch in range(num_epochs):

        beta = min(1, epoch / tau)  # KL-annealing

        # DISCRIMINATOR
        for _ in range(disc_cycles):
            inputs, labels, pca, train_iter = get_next_batch(train_iter, train_loader)
            inputs = inputs.to(device)
            labels = labels.to(device)
            pca = pca.to(device)

            disc_opt.zero_grad()

            cur_bs = inputs.size(0)

            if mode == "PCA":
                z = torch.randn(cur_bs, 100).to(device)
                fake_data = gen(z, labels, pca)  # (B, 1, C)
                real_data = inputs.unsqueeze(1)  # (B, C) -> (B, 1, C)
                with torch.amp.autocast(device):
                    d_loss = disc_loss(real_data, fake_data, disc, gp_weight, device)
                scaler.scale(d_loss).backward()
                scaler.step(disc_opt)
                scaler.update()
            elif mode == "VAE":
                real_data = inputs.unsqueeze(1)  # (B, 1, C)
                mu, log_s = enc(real_data)
                z = reparameterize(mu, log_s)
                fake_data = gen(z.detach(), labels, None)  # (B, 1, C)
                with torch.amp.autocast(device):
                    d_loss = disc_loss(real_data, fake_data, disc, gp_weight, device)
                scaler.scale(d_loss).backward()
                scaler.step(disc_opt)
                scaler.update()

        # CLASSIFIER
        inputs, labels, _, train_iter = get_next_batch(train_iter, train_loader)
        inputs = inputs.to(device)
        labels = labels.squeeze(1)  # (B,)
        labels = labels.to(device)

        cls_opt.zero_grad()

        real_data = inputs.unsqueeze(1)  # (B, C) -> (B, 1, C)
        scores = cls(real_data)  # (B, num_classes)

        with torch.amp.autocast(device):
            cls_loss = class_loss(scores, labels)

        scaler.scale(cls_loss).backward()
        scaler.step(cls_opt)
        scaler.update()

        # GENERATOR / ENCODER
        inputs, labels, pca, train_iter = get_next_batch(train_iter, train_loader)
        inputs = inputs.to(device)
        labels = labels.to(device)
        pca = pca.to(device)

        gen_opt.zero_grad()

        cur_bs = inputs.size(0)

        for p in disc.parameters():
            p.requires_grad = False

        for p in cls.parameters():
            p.requires_grad = False

        if mode == "PCA":
            z = torch.randn(cur_bs, 100).to(device)
            fake_data = gen(z, labels, pca)  # (B, 1, C)

            scores = cls(fake_data)  # (B, num_classes)
            labels = labels.squeeze(1)  # (B,)

            with torch.amp.autocast(device):
                g_gan_loss = gen_loss(fake_data, disc)
                g_cls_loss = class_loss(scores, labels)
                g_loss = g_gan_loss + g_cls_loss

            scaler.scale(g_loss).backward()
            scaler.step(gen_opt)
            scaler.update()

        elif mode == "VAE":

            enc_opt.zero_grad()

            real_data = inputs.unsqueeze(1)  # (B, 1, C)
            mu, log_s = enc(real_data)
            z = reparameterize(mu, log_s)
            fake_data = gen(z, labels, None)  # (B, 1, C)

            scores = cls(fake_data)  # (B, num_classes)
            labels = labels.squeeze(1)  # (B,)

            with torch.amp.autocast(device):
                rec_loss = recon_loss(fake_data, real_data, alpha)
                kl_div_loss = kl_loss(mu, log_s)
                cls_loss = class_loss(scores, labels)
                g_gan_loss = gen_loss(fake_data, disc)
                vae_loss = recon_weight * rec_loss + beta * kl_div_loss + cls_loss + g_gan_loss

            scaler.scale(vae_loss).backward()
            scaler.step(gen_opt)
            scaler.step(enc_opt)
            scaler.update()

        for p in disc.parameters():
            p.requires_grad = True

        for p in cls.parameters():
            p.requires_grad = True

