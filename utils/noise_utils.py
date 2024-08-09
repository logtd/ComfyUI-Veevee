import torch 


def get_alphacumprod(sigma):
    return 1 / ((sigma * sigma) + 1)


def add_noise(src_latent, noise, sigma):
    alphas_cumprod = get_alphacumprod(sigma)

    sqrt_alpha_prod = alphas_cumprod ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(src_latent.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(src_latent.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * src_latent + sqrt_one_minus_alpha_prod * noise
    return noisy_samples


def add_noise_test(latents, sigma, noise=None):
    alpha_cumprod = 1/ ((sigma * sigma) + 1)
    sqrt_alpha_prod = alpha_cumprod ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
    if noise is None:
        generator = torch.Generator(device='cuda')
        # generator.manual_seed(0)
        noise = torch.empty_like(latents).normal_(generator=generator).to(latents.device)

    return sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise