from helper import *

class Houdini(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Y_pred, Y, task_loss, ignore_index=255):

        max_preds, max_inds = Y_pred.max(axis=1)

        mask            = (Y != ignore_index)
        Y               = torch.where(mask, Y, torch.zeros_like(Y).to(Y.device))
        true_preds      = torch.gather(Y_pred, 1, Y).squeeze(1)

        normal_dist     = torch.distributions.Normal(0.0, 1.0)
        probs           = 1.0 - normal_dist.cdf(true_preds - max_preds)
        loss            = torch.sum(probs * task_loss.squeeze(1) * mask.squeeze(1)) / torch.sum(mask.float())

        ctx.save_for_backward(Y_pred, Y, mask, max_preds, max_inds, true_preds, task_loss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):

        Y_pred, Y, mask, max_preds, max_inds, true_preds, task_loss = ctx.saved_tensors

        C = 1./math.sqrt(2 * math.pi)

        temp        = C * torch.exp(-1.0 * (torch.abs(true_preds - max_preds) ** 2) / 2.0) * task_loss.squeeze(1) * mask.squeeze(1)
        grad_input  = torch.zeros_like(Y_pred).to(Y_pred.device)

        grad_input.scatter_(1, max_inds.unsqueeze(1), temp.unsqueeze(1))
        grad_input.scatter_(1, Y, -1.0 * temp.unsqueeze(1))

        grad_input  /= torch.sum(mask.float())

        return (grad_output * grad_input, None, None, None)

