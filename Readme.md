# kaggle-favorita

This code achieves a top 3% result in the Kaggle Corporacion Favorita Grocery Sales Forecasting competition
(https://www.kaggle.com/c/favorita-grocery-sales-forecasting).

## Instructions for running the code

Run full_v2.py to generate the intermediate results, then run blend_3.py to blend the outputs of different models
 and make the submission file.
The script blend_3.py can be run in test mode (for validating the model) or non-test (to actually generate the submission).

## Approach

The final submission is a blend of 3 models: 3 LightGBM boosted regression tree models, developed by me with
a fairly rich set of engineered features; and 2 LightGBM models based on public scripts. I used the 15-day validation period
1/8/2017 to 15/8/2015, and selected models based on their performance on the last 10 days of this period (since the
public leaderboard was based on the first 5 days https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/44962).

I didn't use the oil prices or any external data (I downloaded some weather data from
https://www.ncdc.noaa.gov/cdo-web/datasets, but it was very patchy, so I didn't think it would be useful). I did use
the store transaction data, but I don't think this added much.

Towards the end of the competition, I began running the models in the AWS EC2 cloud, since my laptop couldn't handle the
volume of data required for a 2-year training period.


## My LightGBM models

I trained the same LightGBM model on 3 different training periods, each time computing the features only on that
training period. The longest training period gave the best validation score, but the others contributed to the ensemble
and improved the overall score.

Since the training data only includes store/item/day combinations where net sales were non-zero, it is necessary to
add back the zero rows so that the model can correctly learn the probabilities. This requires some care though; for
example, we need to try to avoid adding zero rows where the product is not actually stocked in the store. So, I only
added rows for the period after the first appearance of a particular item at a particular store. I set onpromotion to
zero for these added rows (using the simplifying assumption that a product on promotion is likely to have at least 1 sale).

I excluded from training items and store combinations for which the last appearance was before 15/1/2017, since these
products are not representative of the product mix in the test set.

I excluded dates around the Christmas/new year period, and near the 2016 earthquake, since these periods are likely
to be unrepresentative of the test set.

I calculated several "group average" features, for example the average sales by store and day of week, item and day of week,
etc. Instead of using the simple average, I actually calculated a Bayesian estimate of the true mean (with priors given
by a broader group), so that for groups with a small sample size the estimate is closer to the assumed prior. This reduces noise
in the feature. Additionally, I excluded the current row from the average calculations, to avoid overfitting the training set.
Finally, onpromotion days and holiday days were excluded when calculating the average.

To reduce correlation between features, I calculated some features as differences from others. For example, the
"average sales by item and day of week" feature is actually the difference between average sales for each item and day,
and overall average sales for the item.

I calculated a "promotion factor" for each item and store, representing the amount by which we expect sales to increase
when the item is on promotion. I similarly calculated a "holiday factor". Both of these are actually differences, not factors
(that is, calculated by subtraction, not division). When working with the raw sales figures, before the log transform,
I would expect the effect of promotion would be multiplicative, but after the log transform, it would become additive.

I also included day of week and day of month features, as well as the number of days since the last promotion
and days to the next promotion.

I included 3 rolling average features: an average of the sales for each item and store over the last 7, 14, and 28
days. For the test period, I simply projected forward these values as they were on the first day of the test period.
Surprisingly, this approach gave quite good results over the validation period. I tried several more sophisticated approaches,
(including one similar to that used in the public script, training a different model for each day of the test period)
but none outperformed this basic one. Moreover, the performance of my three models in the initial part of
the validation period is much better that that of the public script, even though the public script includes more recent
time series features. However, for both by models and the public script, performance decreases towards the end of the
validation period.


## Public scripts

I had 2 models based on public scripts. One was essentially the same as https://www.kaggle.com/tunguz/lgbm-one-step-ahead?scriptVersionId=1993971.
The other was similar, but with a
longer training period and fewer features. Both are generated by the module public_scripts.py. This second version
had lower validation score, but improved the overall score when included in the ensemble.

I did make a couple of changes to the logic of the public scripts:  I didn't use early stopping, since I thought this was likely to
cause overfitting, and removed the arbitrary cap on the size of the predicted value. Apart from this,
I left the structure of the code mostly unchanged, except for modifying it to write the output in a
format suitable for my blending scripts,
and changing the validation period to the one I was using. I also didn't attempt to clean up the formatting of the
public scripts or align it with my own preferred style.


## Blending

My two final submissions were both comprised of the same models, but with different weightings. The best-scoring one
have equal weights for each of the 5 models, and the other (which gave better validation scores) gave higher weights
to the better-performing models.