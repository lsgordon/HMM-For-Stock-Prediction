<<<<<<< HEAD
Initial Commit
=======
# Lab notebook
## 11/9/24 2:00 pm
- setting up the github repository
- making the first starter files for the project
## 11/10/24 12:00 pm
- putting some stuff in the init method
- wrangling data because I can't actually start developing the class until I have some sort of df
- found decile of log returns
## 11/26/24 2:30 pm - Lab session
- Building the forward backward algorithm
- Debugging the forward backward algorithm
## 12/5/24 2:30
- finishing the baum welch
- forgot to do lab notebook, will look at commit history to determine
## 12/6/24 4:30-6:30
- finished the visualization component of the lab
- normalized the probabilities of the lab

## 12/06/24 6:42 pm
- developing the clean_script.py file to clean the stock data for lin_reg
- imported original stock data of lin_reg and initial clean (to be refined) of original
- beginning update of lab notebook for earlier entries

## 12/06/24 11:00 pm
- drafting presentation for 12/09
- adding non-timestamped additions to lab notebook

## 12/08/24 12:00pm-11:45pm
- finished debugging clean_script.py and cleaning stock data
- ported regression code from lab3, obtained weights
- plotted predictions vs actual
- added plots to figures

## 12/09/24 12:45am
- identified most significant outlier in results: NRG.N, energy management company

## 12/10/24
- discussed future steps for the project
- figuring out automatic model validation

## 12/18/24
- implemented said model validation
- found that the model slightly outperforms baseline this is not significantly significant, and for it to be, we would need about 10 years of training data:
We can calculate the mean absolute error of the discrete uniform distribution ($a = 1,b=10$), and we find that the best constant is 5.


Our goal therefore, is to show that our model beats this constant in a statistically significant way, using the CLT


$$\frac{\bar{x}-\mu}{\sigma/\sqrt{n}}$$


we can substitue in for the properties of this distribution to find


$$\frac{\bar{x}-2.5}{2.872/\sqrt{n}}$$


which we would want to give us a test statistic larger than about $1.67$, to be significant at the 95% confidence level.


When we tune the hyperparameters for the baum-welch, we can minimize $\bar{x}$ to about 2.35, but we start to overfit after about 15 iterations, which brings the training MAE back up to about 2.5. This gives us a numerator of about $.15$ in that CLT equation. The other thing we could try is to make $n$ so large that it forces the bottom part of the fraction so large, and this might be an interesting idea for a backtest.


It does still, however, probably outperform the market ever so slightly.
>>>>>>> 86cfb90 (Updates on data)
