+++
title = "Partial Pooling for Estimating Interaction Effects in Factor Models" 
date = "2025-11-08" 
description = "Partial Pooling for Estimating Interaction Effects in Factor Models" 
[taxonomies] 
"writing/tags" = ["methods"] 
+++

# Introduction to the Problem  
A few weeks ago I woke up to this tweet by Gappy:

<blockquote class="twitter-tweet">
  <p lang="en" dir="ltr">I am thinking of writing a note full of spite and thunder about the practice of intersecting stock characteristic (i.e., let us select stocks that are high mtmo, high profitability, and low short interest). It’s a popular practice. You should not do it. I’ll probably restrict circulation of the note.</p>
  &mdash; Gappy (Giuseppe Paleologo) (@__paleologo) <a href="https://twitter.com/__paleologo/status/1979212187202683372">Oct 17, 2025</a>
</blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Gappy is currently the head of quantitative research at BAM, known also for his work in gardening and his books on portfolio management. His day job involves managing portfolio managers. Most likely, some portfolio manager naively intersected stock characteristics and had to deal with the consequences.

Stocks living at the intersection of different characteristics supposedly have abnormal returns. The signal is hard to isolate, and naive overfitting and overconfidence is a quick killer. Financial data has low signal-to-noise, and we cannot afford to shrink the dataset so much if we'd like to avoid overfitting our strategies. If half of the stocks on the market have high mtmo, half of those have high profitability, and half of those had low short interest, then the effective size of the dataset gets cut by a factor of 8. For n factors, that cuts the dataset by an average factor of $2^n$. 

We want to estimate the abnormality of the returns without shrinking our dataset too much. There's a way out of this mess. There are many more stocks that share $n-1$ (out of $n$) of the characteristics we care about, and even more which share only $n-2$. If we can find a way to use these near-miss stocks to raise our effective sample size, we may be able to sufficiently isolate the signals we care about. Before we dive into our technique for estimating these signals more effectively, we should first talk about what we mean by the signals we're trying to isolate.


# What Do We Mean by Abnormal Returns, Anyway?

In recent years, any portfolio manager or researcher, quantitative or not, has come to view market returns through the lens of factor models. They've become the default language for defining risk and, more importantly, for identifying what's left over: "alpha" or abnormal returns. We will use this lens as well.

## Factor Models
It was Gappy himself who wrote the book I used to learn about factor models and to develop an interest in finance, *Elements of Quantitative Investing* (EQI). If you've ever looked at how the prices of AAPL, GOOGL, or MSFT changed throughout the past day, month, or year, you'd probably notice that the paths they take look like deformations of one another. If we look at stocks belonging to a certain industry, say energy or tech, we'd notice that they move along similar paths as well. 

The fundamental idea behind factor modelling is that the true building blocks of the stock market are not the individual equities themselves, but rather deeper, fundamental questions, factors, driving the future. These questions may be along the lines of "Will tariff rates cause trade to stop?" or "Will AI become God?". 

The answers to these questions affect more than just one equity. If we bought shares of two international shipping companies, we may have diversified ourselves to the idiosyncrasies of each company, but we certainly haven't diversified our exposure to the question of tariffs.

More formally, suppose we have a daily sequence of stock prices of $n$ stocks $\textbf{p}_t$. Each $p^i_t$ represents the price of stock $i$ at time $t$. As a convenience, assume that the risk-free rate is 0, then to get the (excess) log returns , we take $\textbf{r}_t = \ln \textbf{p}_t - \ln \textbf{p}\_{t-1}$. Factor models make the decomposition
$$
\mathbf{r}_t = \boldsymbol{\alpha} + \mathbf{B} \mathbf{f}_t + \boldsymbol{\epsilon}_t,
$$
where, following the book EQI,

* $\boldsymbol{\alpha}$ is an $n$-dimensional vector representing the mean excess return of each stock.
* $\mathbf{f}_t$ is a mean $0$, $m\ll n$ dimensional random vector with a covariance matrix $\boldsymbol{\Omega}^f$ which represents the, factors, updates to the underlying questions facing the market.
* The $n\times m$ dimensional matrix $\mathbf{B}$ represents the loading of the effects of the $\mathbf{f}_t$  updates onto $\mathbf{r}_t$.
* Most importantly, $\boldsymbol{\epsilon}_t$ is a random mean 0 vector of with a diagonal covariance matrix $\boldsymbol{\Omega}^\epsilon$ representing the randomness in idiosyncratic returns.

The model implies a return covariance of $\boldsymbol{\Omega}^r=\mathbf{B}\boldsymbol{\Omega}^f\mathbf{B}^{\top}+\boldsymbol{\Omega}^\epsilon$. The coordinates of $\mathbf{B}$ are usually referred to in another manner. If we write $\mathbf{B}=[\beta\_{i,j}]$, then we would say that stock $i$ has a high positive beta exposure to factor $j$ if $\beta_{i,j}$, the $i,j$'th coordinate of $\mathbf{B}$ is high, *mutatis mutandis*. If "high beta" is used without context, it refers to exposure to "the market", a factor commonly used in these factor models.
### Decomposition of alpha

The vector $\boldsymbol{\alpha}$ can be further decomposed into **alpha spanned** $\boldsymbol{\alpha}\_{\parallel}$ and **alpha orthogonal** $\boldsymbol{\alpha}\_{\perp}$. The spanned component, $\boldsymbol{\alpha}\_{\parallel}$, lies in the column space of $\mathbf{B}$ and can be written as
$$
\boldsymbol{\alpha}\_{\parallel} = \mathbf{B} \mathbf{c},
$$
for some vector $\mathbf{c}$ of factor premia. This component represents excess returns that can be explained as compensation for systematic risk,the kind of “expected alpha” that rational investors might demand for bearing exposure to common sources of uncertainty.

The orthogonal component, $\boldsymbol{\alpha}\_{\perp}$, satisfies $\mathbf{B}^\top \boldsymbol{\alpha}\_{\perp} = 0$, meaning it cannot be replicated by any linear combination of the factors. This is what we refer to as abnormal returns. When we say a stock has “alpha,” we are really referring to its $\boldsymbol{\alpha}\_{\perp}$ component. Finding alpha is the equivalent of striking gold in the investing world.

### Choice of Factors
The loading matrix $\mathbf{B}$ and choice of factor distribution is not well defined in terms of the distribution of our returns. For any invertible matrix $\mathbf{C}$, we may define $\tilde{\mathbf{B}}=\mathbf{B}\mathbf{C}$ and $\tilde{\mathbf{f}}_t=\mathbf{C}^{-1}\mathbf{f}_t$, in which case we still get the same decomposition of $\mathbf{r}_t$ and $\boldsymbol{\alpha}$ as before, just with different individual factor components and a new factor covariance $\tilde{\boldsymbol{\Omega}}^f=C^{-1}\boldsymbol{\Omega}^f(C^{-1})^{\top}$. It's always possible find an equivalent factor model in which $\boldsymbol{\Omega}^f=I$ by choosing $C$ so that the individual factors are uncorrelated with a unit variance.

In most major vendor models, such as Barra factor models, there is a factor for just about anything. I've heard complaints that they add new factors to justify their subscription more than anything else.

When we later implement a crude factor model to test out the estimation technique, the first factor will be the daily return of the SP500, and the next $m-1$ factors will be principal components of stock returns with exposure to SP500 removed, so that each factor is uncorrelated and all but the first have unit variance.

## Abnormal Returns In Our Problem

Factor models give a linear decomposition of stock returns, and that linearity imposes an effective independence assumption. When we look for abnormal returns due to interactions between factors, we are looking for returns arising from where this assumption breaks down. Going back to the original tweet, maybe stocks with high mtmo (medium term monthly momentum), high profitability, and high short interest have positive (or negative) alpha due to nonlinearities in expected returns.

We are looking to estimate the average $\boldsymbol{\alpha}\_{\perp}$ for stocks that are high in some combination of factors. By high, we mean they fall above the 70th percentile, arbitrarily chosen, of beta exposure to each of our desired factors. We are **not** looking to estimate $\boldsymbol{\alpha}\_{\parallel}$, except as needed to isolate the effects of $\boldsymbol{\alpha}\_{\perp}$.

Later, when we implement the techniques described here in code, we will build a crude factor model to estimate the $\boldsymbol{\alpha}\_{\perp}$ of each stock ourselves. In this post, rather than focusing on creating the initial estimates of each stock, we focus on combining them together to combat the problem posed in the tweet. We will assume that for each stock, we have independent estimates of $\boldsymbol{\alpha}\_{\perp}$ with a known estimation uncertainty. Now that we know what we're trying to achieve, we can look at how to get there with a concept called partial pooling.

# Partial Pooling

The main task at hand is to combine the information provided by near-miss stocks, the kind which satisfy most, but not all, of the criteria we are looking for. We don't want to treat the near misses exactly as if they were a match, since that muddies the signal we're trying to find. We also may believe that the near misses may contain a signal that isn't so far off. Instead of estimating each intersection separately, move our matched estimate towards the near-miss estimate.

A similar problem has been popularized by Andrew Gelman known as the 8 Schools Problem.

## The 8 Schools Problem

The problem is simple: We have 8 different schools. We run a small SAT coaching program in each one and want to estimate the "true" effect (the average SAT score boost) of the program in each school. For each school $j$, we get an independent, noisy estimate $y_j$ of the effect $\theta_j$, with a known estimation error $\sigma_j$, found by using the standard mean-variance estimators for each school. Using the aggregated information from each school, we try to come up with estimates $\widehat{\theta}_j$.


*Table 1: Estimated Effect of the Program*
| School | $y_j$ | $\sigma_j$ |
| ------ | ----  | ---------  |
| 1      | 28    | 15         |
| 2      | 8     | 10         |
| 3      | -3    | 16         |
| 4      | 7     | 11         |
| 5      | -1    | 9          |
| 6      | 1     | 11         |
| 7      | 18    | 10         |
| 8      | 12    | 18         |

The standard errors in the aggregates are pretty large, but each individual estimate is unbiased. We can stop and say that we're done and use $\widehat\theta_j=y_j$. This is the extreme individualist view. It's extremely ambitious and does its best to estimate each individual effect, and if there were infinite data, it would find the signal exactly. Unfortunately, there isn't an infinite amount of data, and this method falls short of what it tries to do. The other extreme, pure collectivism, is to fully pool the data together, giving an estimate
$$
\hat{\theta}^{\text{pooled}}=\frac{\sum_{i=1}^8\frac{y_i}{\sigma^2_i}}{\sum_{i=1}^8\frac{1}{\sigma^2_i}}=7.68,
$$
with a standard error of
$$
\sigma =\sqrt{\frac{1}{\sum_j 1/\sigma_j^2}}
= 4.07.
$$
The standard error is a lot smaller than any of the standard errors before, but it comes at the cost of us giving up most of our ambition. Full pooling only tries to estimate the average effect across all schools, individual differences be damned. In its humility, it can provide us an estimate for the effect size of each school given only one sample.

In my view, the core art of practicing statistics is to accept the right level of ambition afforded to us by our dataset. The partial pooling technique provides a noble attempt towards this value.

## Partial Pooling the 8 schools

Our data tells us what we need to know. If the individual estimates are far apart, beyond the ambiguity presented by the standard errors, collectivism presents too large a cost. We shouldn't pool much, if at all. If the estimates are close together, so close that we can't tell if there are differences at all, pooling becomes a better option.

The partial pooling estimator assumes the individual effects come from a normal distribution whose parameters we estimate. If it's wide compared to the standard errors of a particular school, we keep our ambition and pool its estimate only slightly. If thin, we take a more humble approach.

More formally, we assume each school’s “true” effect $\theta_j$ is drawn from a common distribution,
$$
\theta_j \sim \mathcal{N}(\mu, \tau^2), \qquad y_j \mid \theta_j \sim \mathcal{N}(\theta_j, \sigma_j^2).
$$
Here $\mu$ is the shared population mean and $\tau^2$ captures how much schools truly differ from one another.

The posterior mean for each school becomes a weighted average between the individual estimate and the shared mean:
$$
\widehat{\theta}_j = (1 - w_j) y_j + w_j \widehat{\mu},
\quad
w_j = \frac{\sigma_j^2}{\sigma_j^2 + \widehat{\tau}^2}.
$$

Qualitatively, this gives us what we want. Quantitatively, it's always possible to tinker and improve, but this works for now.

A true Bayesian would impose a prior on $\mu$ and $\tau$. I am not a true Bayesian. $\mu$ and $\tau$ can be estimated from the data, with the most common technique to be via maximum likelihood, assuming normally distributed errors. 

The maximum likelihood estimate of each is:
$$
\mu^{\text{MLE}} = \frac{\sum_j \frac{y_j}{(\tau^2)^{\text{MLE}} + \sigma_j^2}}{\sum_j \frac{1}{(\tau^2)^{\text{MLE}} + \sigma_j^2}},
\qquad
(\widehat{\tau}^2)^{\text{MLE}} = \max\left(0,
\frac{\sum_j \frac{(y_j - \mu^{\text{MLE}})^2}{\sigma_j^2} - (n - 1)}
{\sum_j \frac{1}{\sigma_j^2}}
\right).
$$

As the value of $\widehat{\tau}^{\text{MLE}}$ increases, $\widehat{\mu}^{\text{MLE}}$ shifts from the fully pooled estimate to the unweighted sample mean. A large $\tau$ corresponds to each school's effect becoming a more noisy signal of all schools as a whole, including those not in the study. When we study interaction effects, we don't care about hypothetical stocks that don't exist. Our sample is the population. If we knew the signal of each stock perfectly, we would also know the true population mean. For this reason and for simplicity, we use as estimators

$$
\widehat{\mu} = \hat\theta^{\text{pooled}},
\qquad
\widehat{\tau}^2 = \max\left(0,
\frac{\sum_j \frac{(y_j - \widehat{\mu})^2}{\sigma_j^2} - (n - 1)}
{\sum_j \frac{1}{\sigma_j^2}}
\right).
$$

With the 8 school data, this is
$$
\hat{\mu}=7.68,\quad \hat{\tau} = 0.
$$
The individual estimates aren't far enough for any individuality, so we totally pool our estimates and our final estimate for each school is $\hat{\theta}_i=7.68$ for each $i$. Out of curiosity, I checked what these MLE parameters would be if I halved the standard errors of each individual estimate, and I got $\hat{\mu}=7.68$ and $\hat{\tau}=7$. If we've had more precise data, we'd have been able to estimate more individuality.

Bayesian methods are good for estimating uncertainty. We can treat $\widehat{\mu}$ as an observation of $\theta_i$ with a noise of variance $\tau^2+\Sigma_\mu$ where

$$
\Sigma_\mu=\text{Var}(\mu)=\frac{1}{\sum_j \frac{1}{\sigma_j^2}}.
$$

$y_i$ and $\widehat{\mu}$ aren't independent, since $\widehat{\mu}$ depends mechanically on $y_i$, but if we treat them as independent, the uncertainty variance of $\hat{\theta}_i$ is

$$
\Sigma\_{i}=\frac{1}{\frac{1}{\sigma_i^2}+\frac{1}{\tau^2+\Sigma_\mu}}
$$



With partial pooling as a building block, we can now turn towards our interaction problem.


# Interaction Partial Pooling

The same principle we used for 8 schools applies to our factor interaction problem: instead of schools, we now have groups of stocks defined by combinations of factor exposures. Unlike the 8 schools, estimating interaction alphas will involve more steps, and we'll move one step at a time. Before we estimate interactions between all $n$ factors, we'll estimate interactions between all combinations of $n-1$ of them. Before that, we'll do $n-2$ and so on, all starting from estimating the effect of having high beta exposure to one factor at a time.


To start, we'll define terms. We can define group (1) to be the collection of stocks high (>70% percentile) in factor 1, (2) stocks high in factor 2, etc. We can then define groups like (1,2,3,4) as the intersection of groups (1), (2), (3), and (4). The initial estimates for each are quite easy to find. For the initial individual estimates for each group, we fully pool the data within:


$$
y\_{(1)}=\frac{\sum_{i\in (1)}\frac{y_i}{\sigma_i^2}}{ \sum_{i\in (1)}\frac{1}{\sigma_i^2}} \quad \sigma\_{(1)}^2 =\frac{1}{\sum_{i\in (1)} \frac{1}{\sigma_j^2}},
$$
and similarly for the other groups. 

<svg width="560" height="300" viewBox="0 0 560 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Partial pooling diagram">
  <!-- Arrowhead -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>

  <!-- Top node -->
  <circle cx="280" cy="60" r="36" fill="#ffffff" stroke="#666" stroke-width="2"/>
  <text x="280" y="65" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#222">μ, τ</text>

  <!-- Bottom nodes -->
  <circle cx="110" cy="220" r="26" fill="#ffffff" stroke="#666" stroke-width="2"/>
  <text x="110" y="225" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">(1)</text>

  <circle cx="230" cy="220" r="26" fill="#ffffff" stroke="#666" stroke-width="2"/>
  <text x="230" y="225" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">(2)</text>

  <circle cx="330" cy="220" r="26" fill="#ffffff" stroke="#666" stroke-width="2"/>
  <text x="330" y="225" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">(3)</text>

  <circle cx="450" cy="220" r="26" fill="#ffffff" stroke="#666" stroke-width="2"/>
  <text x="450" y="225" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">(n)</text>

  <!-- Arrows -->
  <line x1="266" y1="96" x2="122" y2="194" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="274" y1="96" x2="238" y2="194" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="286" y1="96" x2="322" y2="194" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="294" y1="96" x2="438" y2="194" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
</svg>



To get updated estimates, we just use the standard pooling described in the previous section, where groups (1), (2), ..., (n) take the place of our schools.


<div style="max-width:720px;margin:2rem auto;">
  <svg viewBox="0 0 960 420" width="100%" xmlns="http://www.w3.org/2000/svg"
       role="img" aria-label="Factor interaction partial pooling (level-2)">
    <title>Factor interaction partial pooling (level-2)</title>

  <!-- TOP NODES -->
  <circle cx="300" cy="90" r="46" style="fill:#fff;stroke:#666;stroke-width:2;stroke-linecap:round;stroke-linejoin:round"/>
  <text x="300" y="96" text-anchor="middle"
        style="font:600 18px/1.1 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#222">(1)</text>

  <circle cx="660" cy="90" r="46" style="fill:#fff;stroke:#666;stroke-width:2;stroke-linecap:round;stroke-linejoin:round"/>
  <text x="660" y="96" text-anchor="middle"
        style="font:600 18px/1.1 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#222">(n)</text>

  <!-- BOTTOM ROW -->
  <circle cx="180" cy="320" r="34" style="fill:#fff;stroke:#666;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round"/>
  <text x="180" y="326" text-anchor="middle"
        style="font:500 16px/1.1 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#222">(1,2)</text>

  <circle cx="300" cy="320" r="34" style="fill:#fff;stroke:#666;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round"/>
  <text x="300" y="326" text-anchor="middle"
        style="font:500 16px/1.1 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#222">(1,3)</text>

  <circle cx="480" cy="320" r="34" style="fill:#fff;stroke:#666;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round"/>
  <text x="480" y="326" text-anchor="middle"
        style="font:500 16px/1.1 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#222">(1,n)</text>

  <circle cx="660" cy="320" r="34" style="fill:#fff;stroke:#666;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round"/>
  <text x="660" y="326" text-anchor="middle"
        style="font:500 16px/1.1 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#222">(2,n)</text>

  <circle cx="840" cy="320" r="34" style="fill:#fff;stroke:#666;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round"/>
  <text x="840" y="326" text-anchor="middle"
        style="font:500 16px/1.1 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#222">(n−1,n)</text>

  <!-- ARROWS -->
  <g style="stroke:#666;fill:#666">
  <!-- left fan (mirrored of right) -->
  <line x1="280" y1="136" x2="180" y2="286" style="stroke-width:2.25;stroke-linecap:round"/>
  <polygon points="180,286 190.1513,281.3148 182.3426,275.0678"/>

  <line x1="300" y1="136" x2="300" y2="286" style="stroke-width:2.25;stroke-linecap:round"/>
  <polygon points="300,286 305,276 295,276"/>

  <line x1="320" y1="136" x2="480" y2="286" style="stroke-width:2.25;stroke-linecap:round"/>
  <polygon points="480,286 476.1243,275.5129 469.2849,282.8083"/>

  <!-- right fan -->
  <line x1="640" y1="136" x2="480" y2="286" style="stroke-width:2.25;stroke-linecap:round"/>
  <polygon points="480,286 490.7151,282.8083 483.8757,275.5129"/>

  <line x1="660" y1="136" x2="660" y2="286" style="stroke-width:2.25;stroke-linecap:round"/>
  <polygon points="660,286 665,276 655,276"/>

  <line x1="680" y1="136" x2="840" y2="286" style="stroke-width:2.25;stroke-linecap:round"/>
  <polygon points="840,286 836.1243,275.5129 829.2849,282.8083"/>
  </g>
  </svg>
</div>

When we go down another level to try to estimate $\theta\_{(1,n)}$, we end up running into a new issue. We partially pool $y\_{(1,n)}$ with every group below (1), but also separately with the all the groups below (n). We have two competing "priors" for $\theta\_{(1,n)}$. One is that as $N(\hat{\theta}\_{(1)},\tau\_{(1)}^2)$, and the other as $N(\hat{\theta}\_{(n)},\tau\_{(n)}^2)$. 

The two priors are essentially two separate observations of $\theta\_{(1,n)}$, with $\hat{\theta}\_{(1)}$ as an observation of $\theta\_{(1,n)}$ with some $\tau\_{(1)}^2+\sigma^2\_{(1)}$ variance noise. $\Sigma\_{(1)}$ is the Bayesian posterior variance, or the variance of $\theta\_{(1)}$ given $\hat{\theta}\_{(1)}$. $\hat{\theta}\_{(1,n)}$ is treated in the same way. The new question is where to put these observations on the scale of being independent and being effectively one sample.

This isn't something that can be estimated from the initial estimate data. Trying to do something like looking for correlations between parent values across children wouldn't work because any signal would present itself a correlation. 

The stocks in the shared child of any pair of parents show up twice in the pair. If a child has $n$ parents, then the effective parent size $n\_{\text{eff}}$ can be defined number of unique stocks in the parents divided by the sum of counted stocks for each of the parents. To take into account statistical strength, each stock should be weighted by the inverse variance of the estimate. More formally, let $P = \text{parents(group)}$, $U = \bigcup\_{p\in P} p$, then

$$
\frac{n_{\text{eff}}(\text{group})}{n}=\frac{\sum_{i \in U} \frac{1}{\sigma_i^2}}{\sum_{p \in P} \sum_{i \in p} \frac{1}{\sigma_i^2}}.
$$

Weakening the strength of the parents is equivalent to multiplying the variance of their effective noise by $s=n/n\_{\text{eff}}$.

For our estimate of $\hat{\theta}\_{(1,n)}$, we have parents $(1)$ and $(n)$, and we can write
$$
\hat{\theta}\_{(1,n)}=\frac{\frac{y\_{(1,n)}}{\sigma^2\_{(1,n)}} + \frac{1}{s\_{(1,n)}}{(\frac{\hat{\theta}\_{(1)}}{\tau\_{(1)}^2+\Sigma\_{(1)}}+\frac{\hat{\theta}\_{(n)}}{\tau\_{(n)}^2+\Sigma\_{(n)}})}}{\frac{1}{\sigma^2\_{(1,n)}} + \frac{1}{s\_{(1,n)}}{(\frac{1}{\tau\_{(1)}^2+\Sigma\_{(1)}}+\frac{1}{\tau\_{(n)}^2+\Sigma\_{(n)}})}}.
$$
The Bayesian posterior variance $\Sigma\_{(1,n)}$ is

$$
\Sigma\_{(1,n)}=\frac{1}{\frac{1}{\sigma^2\_{(1,n)}} + \frac{1}{s\_{(1,n)}}{(\frac{1}{\tau\_{(1)}^2+\Sigma\_{(1)}}+\frac{1}{\tau\_{(n)}^2+\Sigma\_{(n)}})}}.
$$
By iterating, we have a more appropriate estimate of interaction alphas than we'd have gotten without doing pooling.