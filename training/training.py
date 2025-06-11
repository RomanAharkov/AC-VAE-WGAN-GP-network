from torch import Tensor, nn
from torch.nn import Module

from data.online_generation import generate
from models.models import Generator, Discriminator, Classifier, Encoder
from losses.loss_functions import disc_loss, gen_loss, class_loss, recon_loss, kl_loss
from data.dataset import HyperspectralDataset
import torch
from torch.utils.data import DataLoader, Dataset
from utils.util_functions import get_next_batch, reparameterize, load_models, save_models
import wandb


def logging(d_loss: float, g_loss: float, cls_loss: float, rec_loss: float or None, kl_div_loss: float or None,
            real_batch: Tensor, fake_batch: Tensor):

    wandb.log({'Discriminator loss': d_loss, 'Generator loss': g_loss, 'Classifier loss': cls_loss})
    if rec_loss is not None:
        wandb.log({'Reconstruction loss': rec_loss, 'KL loss': kl_div_loss})

    real_batch = real_batch.cpu().numpy()
    fake_batch = fake_batch.detach().cpu().numpy()

    xs = list(range(real_batch.shape[1]))

    ys_real = real_batch.tolist()
    keys_real = [f"real_{i}" for i in range(real_batch.shape[0])]

    wandb.log({
        "Real spectra":
            wandb.plot.line_series(
                xs=xs,
                ys=ys_real,
                keys=keys_real,
                title="Real spectra",
                xname="Band index")
    })

    ys_fake = fake_batch.tolist()
    keys_fake = [f"fake_{i}" for i in range(fake_batch.shape[0])]

    wandb.log({
        "Fake spectra":
            wandb.plot.line_series(
                xs=xs,
                ys=ys_fake,
                keys=keys_fake,
                title="Fake spectra",
                xname="Band index")
    })


def train_models(num_epochs: int, class_num: int, latent_dim: int or None, channels: int, mode: str,
                 lr_disc: float, lr_gen: float, device: str, train_dataset: HyperspectralDataset, batch_size: int,
                 disc_cycles: int, cls_weight: float, alpha: float or None, tau: int, recon_weight: float or None,
                 gp_weight: float, online_epochs: int, online_passes: int, online_start: int,
                 experiment_load: str or None, load_epoch: int or None, group_name: str, wandb_active=True) -> str:

    assert mode in ("PCA", "VAE"), f"mode must be PCA or VAE, got {mode}"

    # instantiate the models and the data
    gen = Generator(class_num, latent_dim, channels, mode=mode).to(device)
    disc = Discriminator(channels).to(device)
    if mode == "VAE":
        enc = Encoder(latent_dim, channels).to(device)
        enc_opt = torch.optim.Adam(enc.parameters(), lr=lr_gen)
    else:
        enc = None
        enc_opt = None
    cls = Classifier(class_num).to(device)

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.9))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.9))
    cls_opt = torch.optim.Adam(cls.parameters(), lr=lr_disc)

    experiment_name = experiment_load if experiment_load is not None else wandb.sdk.lib.runid.generate_id()

    if wandb_active:
        wandb.login()
        run = wandb.init(
            project=mode,
            id=experiment_name,
            resume="allow",
            group=group_name
        )
        config = wandb.config
        wandb.watch(gen, log_freq=100)
        wandb.watch(disc, log_freq=100)
        wandb.watch(cls, log_freq=100)
        if enc:
            wandb.watch(enc, log_freq=100)

    if experiment_load is not None:
        load_models(experiment_name, disc, disc_opt, gen, gen_opt, cls, cls_opt, enc, enc_opt, load_epoch)
        start_epoch = load_epoch
    else:
        start_epoch = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    train_iter = iter(train_loader)

    for epoch in range(num_epochs):

        beta = min(1.0, epoch / tau)  # KL-annealing

        mean_disc_loss = 0

        # DISCRIMINATOR

        gen.eval()
        if enc is not None:
            enc.eval()

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
                d_loss = disc_loss(real_data, fake_data.detach(), disc, gp_weight, device)
                d_loss.backward()
                disc_opt.step()
            elif mode == "VAE":
                real_data = inputs.unsqueeze(1)  # (B, 1, C)
                mu, log_s = enc(real_data)
                z = reparameterize(mu, log_s)
                fake_data = gen(z.detach(), labels, None)  # (B, 1, C)
                d_loss = disc_loss(real_data, fake_data, disc, gp_weight, device)
                d_loss.backward()
                disc_opt.step()

            mean_disc_loss += d_loss.item() / disc_cycles

        # CLASSIFIER

        inputs, labels, _, train_iter = get_next_batch(train_iter, train_loader)
        inputs = inputs.to(device)
        labels = labels.squeeze(1)  # (B,)
        labels = labels.to(device)

        cls_opt.zero_grad()

        real_data = inputs.unsqueeze(1)  # (B, C) -> (B, 1, C)

        scores = cls(real_data)  # (B, num_classes)
        cls_loss = class_loss(scores, labels)

        cls_loss.backward()
        cls_opt.step()

        # GENERATOR / ENCODER

        disc.eval()
        cls.eval()
        gen.train()
        if enc is not None:
            enc.train()

        for p in disc.parameters():
            p.requires_grad = False

        for p in cls.parameters():
            p.requires_grad = False

        inputs, labels, pca, train_iter = get_next_batch(train_iter, train_loader)
        inputs = inputs.to(device)  # (B, C)
        labels = labels.to(device)
        pca = pca.to(device)

        gen_opt.zero_grad()

        cur_bs = inputs.size(0)

        if mode == "PCA":
            z = torch.randn(cur_bs, 100).to(device)
            fake_data = gen(z, labels, pca)  # (B, 1, C)

            scores = cls(fake_data)  # (B, num_classes)
            labels = labels.squeeze(1)  # (B,)
            g_gan_loss = gen_loss(fake_data, disc)
            g_cls_loss = class_loss(scores, labels)
            g_loss = g_gan_loss + cls_weight * g_cls_loss

            g_loss.backward()
            gen_opt.step()

            logging(mean_disc_loss, g_loss.item(), cls_loss.item(), None, None, inputs,
                    fake_data.squeeze(1))

        elif mode == "VAE":

            enc_opt.zero_grad()

            real_data = inputs.unsqueeze(1)  # (B, 1, C)
            mu, log_s = enc(real_data)
            z = reparameterize(mu, log_s)
            fake_data = gen(z, labels, None)  # (B, 1, C)

            scores = cls(fake_data)  # (B, num_classes)
            labels = labels.squeeze(1)  # (B,)

            rec_loss = recon_loss(fake_data, real_data, alpha)
            kl_div_loss = kl_loss(mu, log_s)
            g_cls_loss = class_loss(scores, labels)
            g_gan_loss = gen_loss(fake_data, disc)
            vae_loss = recon_weight * rec_loss + beta * kl_div_loss + cls_weight * g_cls_loss + g_gan_loss

            vae_loss.backward()
            gen_opt.step()
            enc_opt.step()

            logging(mean_disc_loss, vae_loss.item(), cls_loss.item(), rec_loss.item(), kl_div_loss.item(), inputs,
                    fake_data.squeeze(1))

        if epoch % online_epochs == 0 and epoch >= online_start:
            save_models(experiment_name, disc, disc_opt, gen, gen_opt, cls, cls_opt, enc, enc_opt, start_epoch + epoch)
            generate(gen, train_dataset, online_passes, batch_size, mode, enc, device, experiment_name)

        for p in disc.parameters():
            p.requires_grad = True

        for p in cls.parameters():
            p.requires_grad = True

        disc.train()
        cls.train()

    return experiment_name


def train_cls(train_dataset: Dataset, batch_size: int, model: Module, learning_rate: float, num_epochs: int, device: str
              ) -> Module:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # (B, C) / (B, 1) or (B, class_num)
            if labels.size(dim=1) == 1:
                labels = labels.squeeze(1)  # (B,)
            if len(inputs.shape) != 5:
                inputs = inputs.unsqueeze(1)  # (B, 1, C)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}/{num_epochs}')

    return model
