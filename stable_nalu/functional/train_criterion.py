def get_train_criterion(criterion, mse, pcc, mape, current_iteration, pcc2mse_iteration):
    """

    Args:
        criterion: str: the  type train loss to use
        mse: mse loss
        pcc: pcc loss
        current_iteration: int: current global step/ epoch
        pcc2mse_iteration: int: For use of pcc-mse criterion. Tells you at which iteration to switch from using a pcc
            loss to an mse loss
    Returns: loss values
    """
    if criterion == 'mse':
        return mse
    elif criterion == 'mape':
        return mape
    elif criterion == 'pcc':
        return pcc
    elif criterion == 'pcc-mse':
        return pcc if current_iteration < pcc2mse_iteration else mse
    else:
        raise ValueError(f'{criterion} is not a valid train criterion option')
