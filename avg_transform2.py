"""Functions for MLE and Bayesian estimates of group averages."""

import pandas as pd
import numpy as np

def beta_params(mean, var):
    """Given the mean and variance of a beta distribution, calculates the parameters alpha, beta of this 
    distribution."""

    alpha = mean ** 2 * (1 - mean) / var - mean
    beta = alpha * (1 - mean) / mean
    return alpha, beta


def posterior_mean_beta(mean_prior, var_prior, sample_mean, sample_size):
    """Calculate the posterior Bayesian estimate of the parameter p of a binomial distribution, given the 
    sample mean, assumed variance of the prior, and sample size."""

    alpha_prior, beta_prior = beta_params(mean_prior, var_prior)
    alpha_post = sample_mean * sample_size + alpha_prior
    beta_post = sample_size * (1 - sample_mean) + beta_prior
    return alpha_post / (alpha_post + beta_post)


def posterior_mean_normal(mean_prior, sample_mean, sample_size):
    """Calculate the posterior Bayesian estimate of the parameter p of a normal distribution, given the prior mean,
    sample mean and sample size."""

    return (sample_mean * sample_size + mean_prior) / (sample_size + 1)


def bayesian_group_estimate(df, groupby_col, target_col, dist_type, train_bool=None, exclude_current_row=False,
                            prior_group=None):
    """Group the dataframe by groupby_col, and calculates the Bayesian estimate of the mean of target_col in each group.
    
    Args:
        df: pd.DataFrame
        groupby_col: string, name of column to group by
        target_col: string, name of column to aggregate
        dist_type: string, one of 'beta', 'normal' or 'log_normal'. Use beta where target_col is binary, or normal 
            where target_col is continuous.
        train_bool: pd.Series with same index as df and boolean values specifying which rows should be used to 
            calculate the group average. If None, use all the rows
        exclude_current_row: boolean. If True, exclude the current row when calculating the group average.
        prior_group: list of strings, columns to group by to calculate the prior. Groups should be large enough to be 
            more statistically reliable that the group averages we are trying to estimate.
            
    Returns:
        pd.Series with same index as df, giving the Bayesian estimates of the group average
    """

    # check there are at least 2 groups
    if len(df[groupby_col].drop_duplicates()) < 2:
        raise ValueError('Column {} has only 1 unique value.'.format(groupby_col))

    if train_bool is None:
        train_bool = df.index

    if dist_type == 'log_normal':
        df_copy = df.copy()  # make a copy to avoid renaming existing column
        df_copy['log_target'] = np.log(df[target_col])
        sum_col = df_copy.loc[train_bool].groupby(groupby_col)['log_target'].sum().to_frame('sum')
        size_col = df_copy.loc[train_bool].groupby(groupby_col)['log_target'].size().to_frame('count')
        if type(groupby_col) == str:
            df_col = df[groupby_col].to_frame()
        else:
            df_col = df[groupby_col]
        sample_sum = pd.merge(df_col, sum_col, 'left', left_on=groupby_col, right_index=True)['sum']
        sample_size = pd.merge(df_col, size_col, 'left', left_on=groupby_col, right_index=True)['count']
        if exclude_current_row:
            sample_sum.loc[train_bool] -= df_copy.loc[train_bool, 'log_target']
            sample_size.loc[train_bool] -= 1
        sample_mean = sample_sum / sample_size
        sample_mean.loc[sample_size == 0] = np.nan
        if prior_group is None:
            mean_prior = df.loc[train_bool, 'log_target'].mean()
        else:
            if type(prior_group) == str:
                prior_col = df[prior_group].to_frame()
            else:
                prior_col = df[prior_group]
            sample_mean_prior = df_copy.loc[train_bool].groupby(prior_group)['log_target'].mean().to_frame('mean')
            mean_prior = pd.merge(prior_col, sample_mean_prior, 'left', left_on=prior_group, right_index=True)['mean']
            mean_prior = mean_prior.fillna(mean_prior.mean())
    else:
        sum_col = df.loc[train_bool].groupby(groupby_col)[target_col].sum().to_frame('sum')
        size_col = df.loc[train_bool].groupby(groupby_col)[target_col].size().to_frame('count')
        if type(groupby_col) == str:
            df_col = df[groupby_col].to_frame()
        else:
            df_col = df[groupby_col]
        sample_sum = pd.merge(df_col, sum_col, 'left', left_on=groupby_col, right_index=True)['sum']
        sample_size = pd.merge(df_col, size_col, 'left', left_on=groupby_col, right_index=True)['count']
        if exclude_current_row:
            sample_sum.loc[train_bool] -= df.loc[train_bool, target_col]
            sample_size.loc[train_bool] -= 1
        sample_mean = sample_sum / sample_size
        sample_mean.loc[sample_size == 0] = np.nan
        if prior_group is None:
            mean_prior = df.loc[train_bool, target_col].mean()
        else:
            if type(prior_group) == str:
                prior_col = df[prior_group].to_frame()
            else:
                prior_col = df[prior_group]
            sample_mean_prior = df.loc[train_bool].groupby(prior_group)[target_col].mean().to_frame('mean')
            mean_prior = pd.merge(prior_col, sample_mean_prior, 'left', left_on=prior_group, right_index=True)['mean']
            mean_prior = mean_prior.fillna(mean_prior.mean())

    if dist_type == 'beta':
        # Estimate the prior variance by taking the variance of the groups with >= 100 values (including smaller
        # groups will over-estimate the variance). If there are <= 5 such groups, just use all groups.
        large_ind = sample_size >= 500
        if large_ind.sum() < 5:
            var_prior = sample_mean.var()
        else:
            # Use the variance of the large groups
            var_prior = sample_mean.loc[large_ind].var()
        post_mean = posterior_mean_beta(mean_prior, var_prior, sample_mean, sample_size)
    elif dist_type == 'normal':
        post_mean = posterior_mean_normal(mean_prior, sample_mean, sample_size)
    elif dist_type == 'log_normal':
        post_mean = np.exp(posterior_mean_normal(mean_prior, sample_mean, sample_size))
    else:
        raise ValueError("dist_type must be one of 'beta', 'normal', 'log_normal'")

    post_mean = post_mean.fillna(mean_prior)
    if type(groupby_col) == str:
        col_name = groupby_col + '_group_avg_bayes'
    else:
        col_name = '_'.join(groupby_col) + '_group_avg_bayes'
    post_mean = post_mean.to_frame(col_name)
    return post_mean


def group_estimate(df, groupby_col, target_col, dist_type, train_bool=None, exclude_current_row=False):
    """Group the dataframe by groupby_col, and calculates the maximum likelihood estimate of the mean of target_col in 
    each group.

    Args:
        df: pd.DataFrame
        groupby_col: string, name of column to group by
        target_col: string, name of column to aggregate
        dist_type: string, one of 'beta', 'normal' or 'log_normal'.
        train_bool: pd.Series with same index as df and boolean values specifying which rows should be used to 
            calculate the group average. If None, use all the rows
        exclude_current_row: boolean. If True, exclude the current row when calculating the group average.

    Returns:
        pd.Series with same index as df, giving the estimates of the group average
    """

    # check there are at least 2 groups
    if len(df[groupby_col].drop_duplicates()) < 2:
        raise ValueError('Column {} has only 1 unique value.'.format(groupby_col))

    if train_bool is None:
        train_bool = df.index

    if dist_type == 'log_normal':
        df_copy = df.copy()  # make a copy to avoid renaming existing column
        df_copy['log_target'] = np.log(df[target_col])
        sum_col = df_copy.loc[train_bool].groupby(groupby_col)['log_target'].sum().to_frame('sum')
        size_col = df_copy.loc[train_bool].groupby(groupby_col)['log_target'].size().to_frame('count')
        if type(groupby_col) == str:
            df_col = df[groupby_col].to_frame()
        else:
            df_col = df[groupby_col]
        sample_sum = pd.merge(df_col, sum_col, 'left', left_on=groupby_col, right_index=True)['sum']
        sample_size = pd.merge(df_col, size_col, 'left', left_on=groupby_col, right_index=True)['count']
        if exclude_current_row:
            sample_sum.loc[train_bool] -= df_copy.loc[train_bool, 'log_target']
            sample_size.loc[train_bool] -= 1
        sample_mean = sample_sum / sample_size
        sample_mean.loc[sample_size == 0] = np.nan
        mean_mle = df_copy.loc[train_bool, 'log_target'].mean()
    else:
        sum_col = df.loc[train_bool].groupby(groupby_col)[target_col].sum().to_frame('sum')
        size_col = df.loc[train_bool].groupby(groupby_col)[target_col].size().to_frame('count')
        if type(groupby_col) == str:
            df_col = df[groupby_col].to_frame()
        else:
            df_col = df[groupby_col]
        sample_sum = pd.merge(df_col, sum_col, 'left', left_on=groupby_col, right_index=True)['sum']
        sample_size = pd.merge(df_col, size_col, 'left', left_on=groupby_col, right_index=True)['count']
        if exclude_current_row:
            sample_sum.loc[train_bool] -= df.loc[train_bool, target_col]
            sample_size.loc[train_bool] -= 1
        sample_mean = sample_sum / sample_size
        sample_mean.loc[sample_size == 0] = np.nan
        mean_mle = df.loc[train_bool, target_col].mean()

    if dist_type == 'log_normal':
        mean_mle = np.exp(mean_mle)

    sample_mean = sample_mean.fillna(mean_mle)
    if dist_type == 'log_normal':
        sample_mean = np.exp(sample_mean)
    if type(groupby_col) == str:
        col_name = groupby_col + '_group_avg'
    else:
        col_name = '_'.join(groupby_col) + '_group_avg'
    sample_mean = sample_mean.to_frame(col_name)
    return sample_mean

def group_median(df, groupby_col, target_col, train_bool=None):
    """Group the dataframe by groupby_col, and calculates the median of target_col in each group.

    Args:
        df: pd.DataFrame
        groupby_col: string, name of column to group by
        target_col: string, name of column to aggregate
        train_bool: pd.Series with same index as df and boolean values specifying which rows should be used to 
            calculate the group average. If None, use all the rows

    Returns:
        pd.Series with same index as df, giving the estimates of the group average
    """
    if train_bool is None:
        train_bool = df.index

    std = df.loc[train_bool].groupby(groupby_col)[target_col].std().to_frame('median')
    if type(groupby_col) == str:
        df_col = df[groupby_col].to_frame()
    else:
        df_col = df[groupby_col]
    sample_std = pd.merge(df_col, std, 'left', left_on=groupby_col, right_index=True)['median']

    sample_std = sample_std.fillna(sample_std.mean())
    if type(groupby_col) == str:
        col_name = groupby_col + '_group_median'
    else:
        col_name = '_'.join(groupby_col) + '_group_median'
    sample_std = sample_std.to_frame(col_name)
    return sample_std


def group_std(df, groupby_col, target_col, train_bool=None):
    """Group the dataframe by groupby_col, and calculates the standard deviation of target_col in each group.

    Args:
        df: pd.DataFrame
        groupby_col: string, name of column to group by
        target_col: string, name of column to aggregate
        train_bool: pd.Series with same index as df and boolean values specifying which rows should be used to 
            calculate the group average. If None, use all the rows

    Returns:
        pd.Series with same index as df, giving the estimates of the group average
    """
    if train_bool is None:
        train_bool = df.index

    std = df.loc[train_bool].groupby(groupby_col)[target_col].std().to_frame('std')
    if type(groupby_col) == str:
        df_col = df[groupby_col].to_frame()
    else:
        df_col = df[groupby_col]
    sample_std = pd.merge(df_col, std, 'left', left_on=groupby_col, right_index=True)['std']

    sample_std = sample_std.fillna(sample_std.mean())
    if type(groupby_col) == str:
        col_name = groupby_col + '_group_std'
    else:
        col_name = '_'.join(groupby_col) + '_group_std'
    sample_std = sample_std.to_frame(col_name)
    return sample_std
