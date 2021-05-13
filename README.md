# Multi-factor-dynamic-stock-selection-model--based-on-regression-method
The py file can be run on Joinquant https://www.joinquant.com

This program uses a multi-factor stock selection strategy for empirical analysis, and obtain the historical market data of the constituent stocks of CSI 300 Index (excluding ST shares and all stocks that have been suspended in the past year) by means of the joinquant quantitative platform. Firstly, we obtained 17 factors from valuation data, balance data, financial indicator and cash flow data. Then, a single-factor IC test is performed. 

For each factor, the correlated coefficient between the factor exposure value and the stock return rate in the next period is calculated, and a set of IC sequences, that is, statistics during the period, are obtained. In addition, four indicators were applied to score the IC value, and finally select the factors whose scoring result is greater than 0. These well- selected factors selected are the best performing parts. Moreover, multi- factor regression method was introduced to calculate the weight factor.

The model will automatically select stocks, adjust positions, buy and sell stocks every other week. The process will recalculate every fixed cycle, obtain the best performing factors again, recalculating the new factor weights, and finally select stocks and include them in the stock pool. 

Through this dynamic rotation, the model can continuously change the stock selection strategy in order to achieve the largest excess returns. The multi-factor regression method performed best during the back test process. During the three-year back test period from January 1st, 2017 to May 21, 2020, the strategic return reached 86.07%(the benchmark return :20.79%) and obtaining excess returns of 57.37%. The alpha value is 0.16 and the beta value is 0.97, Sharpe ratio is 0.73, suggesting it is a good strategy.
