import torch

from MODEL.model.coast2 import continuation_loss, continuation_targets


def test_continuation_targets_exclude_current_reward_and_zero_terminal_continuation():
    rewards = [torch.tensor([[-1.0]]), torch.tensor([[-2.0]]), torch.tensor([[-7.0]])]
    alive = [torch.tensor([[True]]), torch.tensor([[True]]), torch.tensor([[True]])]

    targets, valid = continuation_targets(rewards, alive)

    assert targets[:, 0, 0].tolist() == [9.0, 7.0, 0.0]
    assert valid.all()
    assert not targets.requires_grad


def test_continuation_targets_ignore_post_terminal_padding_per_instance():
    rewards = [torch.tensor([[-1.0], [-2.0]]), torch.tensor([[-3.0], [40.0]]), torch.tensor([[50.0], [99.0]])]
    alive = [torch.tensor([[True], [True]]), torch.tensor([[True], [False]]), torch.tensor([[False], [False]])]

    targets, valid = continuation_targets(rewards, alive)

    assert targets[:, :, 0].tolist() == [[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert valid[:, :, 0].tolist() == [[True, True], [True, False], [False, False]]


def test_continuation_loss_is_finite_zero_for_empty_valid_mask():
    prediction = torch.randn(3, 2, 1, requires_grad=True)
    targets = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)

    loss = continuation_loss(prediction, targets, valid)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_continuation_loss_uses_scaled_target_and_only_valid_transitions():
    prediction = torch.tensor([[[1.0]], [[100.0]]], requires_grad=True)
    targets = torch.tensor([[[10.0]], [[-1000.0]]])
    valid = torch.tensor([[[True]], [[False]]])

    loss = continuation_loss(prediction, targets, valid, cost_scale=10.0)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
