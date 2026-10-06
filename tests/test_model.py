import torch

from nerf.model import NeRF


def random_inputs(*shape):
    points = torch.randn(*shape, 3)
    dirs = torch.randn(*shape, 3)
    return points, dirs / dirs.norm(dim=-1, keepdim=True)


def small_model():
    return NeRF(depth=4, width=32, skip=1)


def test_paper_architecture():
    model = NeRF()
    trunk = [(layer.in_features, layer.out_features) for layer in model.trunk]
    # 8 layers of 256; the sixth also receives the 63 encoded-position inputs.
    assert trunk == [(63, 256)] + [(256, 256)] * 4 + [(319, 256)] + [(256, 256)] * 2
    assert (model.sigma_head.in_features, model.sigma_head.out_features) == (256, 1)
    assert (model.feature.in_features, model.feature.out_features) == (256, 256)
    assert (model.dir_layer.in_features, model.dir_layer.out_features) == (256 + 27, 128)
    assert (model.rgb_head.in_features, model.rgb_head.out_features) == (128, 3)
    # What those layer sizes add up to.
    assert sum(p.numel() for p in model.parameters()) == 595_844


def test_output_shapes_and_ranges():
    torch.manual_seed(0)
    points, dirs = random_inputs(6, 10)
    sigma, rgb = small_model()(points, dirs)
    assert sigma.shape == (6, 10)
    assert rgb.shape == (6, 10, 3)
    assert (sigma >= 0).all()
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_density_ignores_view_direction_but_colour_does_not():
    torch.manual_seed(0)
    model = small_model()
    points, dirs_a = random_inputs(200)
    _, dirs_b = random_inputs(200)
    sigma_a, rgb_a = model(points, dirs_a)
    sigma_b, rgb_b = model(points, dirs_b)
    assert torch.equal(sigma_a, sigma_b)
    assert not torch.allclose(rgb_a, rgb_b)


def test_skip_connection_carries_the_position_past_the_early_layers():
    # Zero the layers up to and including the skip layer, so nothing about the
    # position survives them. The output can then only depend on the position
    # through the skip connection.
    torch.manual_seed(0)
    model = small_model()
    with torch.no_grad():
        for layer in model.trunk[: model.skip + 1]:
            layer.weight.zero_()
            layer.bias.zero_()
    points, dirs = random_inputs(50)
    _, rgb = model(points, dirs[:1].expand(50, 3))
    assert rgb.std(dim=0).max() > 1e-4


def test_every_parameter_is_connected_to_the_output():
    torch.manual_seed(0)
    model = small_model()
    points, dirs = random_inputs(500)
    sigma, rgb = model(points, dirs)
    (sigma.sum() + rgb.sum()).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    total = sum(p.grad.abs().sum() for p in model.parameters())
    assert total > 0


def test_same_seed_gives_same_network():
    torch.manual_seed(3)
    a = small_model()
    torch.manual_seed(3)
    b = small_model()
    points, dirs = random_inputs(20)
    sigma_a, rgb_a = a(points, dirs)
    sigma_b, rgb_b = b(points, dirs)
    assert torch.equal(sigma_a, sigma_b) and torch.equal(rgb_a, rgb_b)


def test_initialisation_matches_the_released_code():
    # Glorot uniform: weights within sqrt(6 / (fan_in + fan_out)), zero biases.
    torch.manual_seed(0)
    model = NeRF()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            bound = (6.0 / (module.in_features + module.out_features)) ** 0.5
            assert module.weight.abs().max() <= bound, name
            assert module.weight.abs().max() > 0.9 * bound, name
            assert torch.equal(module.bias, torch.zeros_like(module.bias)), name
