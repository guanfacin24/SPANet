from torch import Tensor
import torch

def disco_loss(
        predictions: Tensor,
        correlation: Tensor,
        weights: Tensor = None,
        scale: float = 1.
):

    ### Mask -1 values
    mask = correlation > 0
    predictions = predictions[mask]
    correlation = correlation[mask]
    weights = weights[mask]

    xx = correlation.view(-1, 1).repeat(1, len(correlation)).view(len(correlation), len(correlation))
    yy = correlation.repeat(len(correlation), 1).view(len(correlation), len(correlation))
    amat = (xx - yy).abs()

    xx = predictions.view(-1, 1).repeat(1, len(predictions)).view(len(predictions), len(predictions))
    yy = predictions.repeat(len(predictions), 1).view(len(predictions), len(predictions))
    bmat = (xx - yy).abs()

    ### TODO: This way, custom weights HAVE to exist
    ### TODO: Could all be set to 1 though
    amatavg = torch.mean(amat * weights, dim = 1)
    Amat = amat - amatavg.repeat(len(correlation), 1).view(len(correlation), len(correlation)) \
           - amatavg.view(-1, 1).repeat(1, len(correlation)).view(len(correlation), len(correlation)) \
           + torch.mean(amatavg * weights)

    bmatavg = torch.mean(bmat * weights, dim = 1)
    Bmat = bmat - bmatavg.repeat(len(predictions), 1).view(len(predictions), len(predictions)) \
           - bmatavg.view(-1, 1).repeat(1, len(predictions)).view(len(predictions), len(predictions)) \
           + torch.mean(bmatavg * weights)

    ABavg = torch.mean(Amat * Bmat * weights, dim = 1)
    AAavg = torch.mean(Amat * Amat * weights, dim = 1)
    BBavg = torch.mean(Bmat * Bmat * weights, dim = 1)

    if scale == 1:
        dCorr = (torch.mean(ABavg * weights)) / torch.sqrt(
            (torch.mean(AAavg * weights) * torch.mean(BBavg * weights)))
    elif scale == 2:
        dCorr = (torch.mean(ABavg * weights)) ** 2 / (
                torch.mean(AAavg * weights) * torch.mean(BBavg * weights))
    else:
        dCorr = ((torch.mean(ABavg * weights)) / torch.sqrt(
            (torch.mean(AAavg * weights) * torch.mean(BBavg * weights)))) ** scale

    return dCorr