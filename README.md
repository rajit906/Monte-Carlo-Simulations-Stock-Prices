# Monte-Carlo-Simulations-Stock-Prices
This code is supplementary to the research project in the link below
https://www.ijser.org/researchpaper/Mathematical-Modeling-in-Finance.pdf
The project uses monte-carlo simulations to predict Apple's stock prices through various statistical tests.

Any variable whose value alters in an uncertain way is claimed to follow a stochastic
process. Reddy & Clinton (2016) say that the concept of stochastic processes is significant in
mathematical finance because it can be utilized to model many phenomena in which the quality
or the factor differs continuously through time (Reddy & Clinton, 2016). Various processes are
always modeled by a stochastic process and are a broad terminology for any assortment of
random variables [𝑋(𝑡)] relying on time t. Time might be discrete for instance, t=1,2 3, or
continuous, t≥0. 

The Brownian motion B (t) is utilized to capture the uncertainty in the future behavior of
stochastic processes and has the subsequent features:
a. (Independence of increments) B(t)-B(s), for t>s, is independent of the past.
b. (Continuity of paths) B (t), t≥ 0, are continuous functions of t.
c. (Normal increments) B(t)-B(s), has Normal distribution with mean 0 and variance t-s, if
𝑠 = 0 then B(t)-B(0)~𝑁(0,𝑡).
Louis Bachelier used Brownian motion to model the prices of stocks. In a distinct form, the
Bachelier model can be written as '''d𝑺𝒕=u𝑺𝒕dt+𝝈𝑺𝒕𝒅𝑾(𝒕)'''
Where S(t) is the price of a stock, 𝐵(𝑡)is the Brownian motion, u is the return on the
price of a stock, 𝜎 is the volatility of the stock price. This equation is known as the arithmetic
Brownian motion. The solution to Equation is d𝑺𝒕=u𝑺𝒕dt+𝝈𝑺𝒕𝒅𝑾(𝒕) (Bae et al., 2015). Let us
first comprehend this definition, normally, u is called the percentage drift and σ is known as
percentage volatility. It is important to consider a Brownian motion course that satisfies this
differential equation. The terminology u𝑺𝒕dt controls the ‘trend’ of this trajectory and the
𝝈𝑺𝒕𝒅𝑾(𝒕) regulates the random noise impact in the trajectory. It is vital to find a solution
because it is a differential equation (Yang, 2015). 
