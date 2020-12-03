import torch
import copy


def test_copy_weights():
    orig_model = torch.nn.Linear(3, 2)
    model2 = torch.nn.Linear(3, 2)
    model3 = torch.nn.Linear(3, 2)

    # NOTE: THIS IS WRONG!!! THIS DOES NOTHING!!!
    for task_p, orig_p in zip(model2.parameters(), orig_model.parameters()):
        assert(not torch.allclose(task_p, orig_p))
        task_p = orig_p.detach()
        assert(torch.allclose(task_p, orig_p))
    for task_p, orig_p in zip(model2.parameters(), orig_model.parameters()):
        assert(not torch.allclose(task_p, orig_p))

    # NOTE: CORRECT WAY TO COPY OVER WEIGHTS
    # copying over weights from model to model2
    # check that assignments in for-loop actually modify model parameters
    for task_p, orig_p in zip(model2.parameters(), orig_model.parameters()):
        assert(not torch.allclose(task_p, orig_p))
        task_p.data = orig_p.data
        assert(torch.allclose(task_p, orig_p))
    for task_p, orig_p in zip(model2.parameters(), orig_model.parameters()):
        assert(torch.allclose(task_p, orig_p))

    # NOTE: ANOTHER CORRECT WAY TO COPY OVER WEIGHTS
    # copying over weights from model to model2
    # check that assignments in for-loop actually modify model parameters
    model3.load_state_dict(orig_model.state_dict())
    for task_p, orig_p in zip(model3.parameters(), orig_model.parameters()):
        assert(torch.allclose(task_p, orig_p))

    # NOTE: ANOTHER CORRECT WAY TO COPY OVER WEIGHTS
    # copying over weights from model to model2
    # check that assignments in for-loop actually modify model parameters
    model4 = copy.deepcopy(orig_model)
    for task_p, orig_p in zip(model4.parameters(), orig_model.parameters()):
        assert(torch.allclose(task_p, orig_p))


def train(model, internal_optimizer):
    x = torch.randn(4, 3)
    x2 = torch.randn(4, 3)
    y = model(x).log_softmax(-1)
    y2 = model(x2)
    loss = torch.mul(y + y2, y2).sum()
    internal_optimizer.zero_grad()
    loss.backward()
    internal_optimizer.step()
    print(loss.item())


def test_copy_grad():
    model = torch.nn.Linear(3, 2)
    internal_optimizer = torch.optim.Adam(model.parameters(), lr=1)

    orig_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(orig_model.parameters(), lr=1)

    train(model, internal_optimizer)

    for task_p, orig_p in zip(model.parameters(), orig_model.parameters()):
        assert(task_p.grad is not None)
        assert(orig_p.grad is None)

    # add up gradients
    sum_gradients = [torch.zeros(p.shape) for p in model.parameters()]
    for sum_grad, src_p in zip(sum_gradients, model.parameters()):
        sum_grad.data += src_p.grad.data

    # apply sum of gradients to original model
    # no need for optimizer.zero_grad() because gradients directly set, not accumulated
    for orig_p, sum_grad in zip(orig_model.parameters(), sum_gradients):
        orig_p.grad = sum_grad

    for task_p, orig_p in zip(model.parameters(), orig_model.parameters()):
        assert(torch.allclose(task_p.grad, orig_p.grad))

    for task_p, orig_p in zip(model.parameters(), orig_model.parameters()):
        assert(not torch.allclose(task_p, orig_p))
    optimizer.step()
    for task_p, orig_p in zip(model.parameters(), orig_model.parameters()):
        assert(torch.allclose(task_p, orig_p))


test_copy_weights()
test_copy_grad()
