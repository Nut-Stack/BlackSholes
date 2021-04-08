import math
import numpy as np
import scipy.stats as stats

#Created by Nut-Stack

e = np.exp(1)
phi = ( 1 + math.sqrt(5) ) / 2
'''
S = current stock price
t = time until option exercise
K = option strike price
r = risk free interest rate
N = Cumulitive standard normal distribution
e = exponential term
s = standard deviation (implied volitility)
ln = natural log
q = annual dividend yield
'''
S = 389.51
t = 9/365
K = 400
#r = 0.0007
r = 0.0016376
s = .1508
q = 0/100

def d1(S,K,r,s,t):
    return(  ((math.log(S/K)) + (r + ((s**2)/2)) * t) / (s * math.sqrt(t)))

def d2(d1,s,t):
    return( d1 - s * (math.sqrt(t)) )

D1 = d1(S,K,r,s,t)
D2 = d2(D1,s,t)


Fair_Value = S * stats.norm.cdf(D1) - K * e**(r * t) * (stats.norm.cdf(D2))
Delta = stats.norm.cdf(D1)
Gamma = (stats.norm.pdf(D1))/(S * s * math.sqrt(t))
Theta = ( (-1*(S * stats.norm.pdf(D1) * s)/(2 * math.sqrt(t))) - r * K * e**(-r * t) * stats.norm.cdf(D2) ) / 365
Vega = ((S) * (stats.norm.pdf(D1)) * (math.sqrt(t)))/(100)
Rho = ((K * t * e**(-r * t) * stats.norm.cdf(D2))/100)
Lambda = (Delta * (S/Fair_Value))/100
Vanna = ((-stats.norm.cdf(D1)) * (D2/s))/100
Charm = (stats.norm.cdf(D1) - (stats.norm.cdf(D1) * ((2 * (r - q) * t) - (D2 * s * - math.sqrt(t))))/(2 * t * s * math.sqrt(t)))/365
Speed = (stats.norm.pdf(D1)) / ((S**2) * s * math.sqrt(t)) * (((D1) / (s * math.sqrt(t))) + 1)
Ultima = ((-Vega)/(s**s)) * (D1 * D2 * (1 - (D1 * D2)) + (D1**2) + (D2**2))
Vomma = Vega * ((D1 * D2)/(s))
Veta =  (-S * Delta * math.sqrt(t) * (q + (((r - q) * D1)/(s * math.sqrt(t))) - (((1 + (D1 * D2)))/(2 * t))))/365
Zomma = Gamma * (((D1 * D2) -1) / s)
POP = stats.norm.cdf(math.log(S/K)/(s * math.sqrt(t)))
Color = (-stats.norm.pdf(D1) / (2 * S * t * s * math.sqrt(t))) * (((2 * q * t) + 1) + ((2 * (r - q) * t) - (D2 * s * math.sqrt(t)))/(s * math.sqrt(t)) * D1)
Epsilon = (-S * stats.norm.cdf(D1))/100

Breakeven_percent = (Fair_Value + K - S) / S
Breakeven_dollars = round((Fair_Value + K - S), 2)
Breakeven_price = S + Breakeven_dollars

'''
first order: Delta, Vega, Theta, Rho, Lambda, Epsilon
second order: Gamma, Vanna, Charm, Vomma, Veta, Vera
third order: Speed, Zomma, Color, Ultima
'''

print(">First Order-----------------------<")
print("Delta 'Δ'  : {:.10f}".format(Delta))
print("Vega       : {:.10f}".format(Vega))
print("Theta 'Θ'  : {:.10f}".format(Theta))
print("Rho 'Ρ'    : {:.10f}".format(Rho))
print("Epsilon 'Є': {:.10f}".format(Epsilon))
print("Lambda 'λ' : {:.10f}".format(Lambda))

print(">Second Order----------------------<")
print("Gamma 'Γ'  : {:.10f}".format(Gamma))
print("Vanna      : {:.10f}".format(Vanna))
print("Charm      : {:.10f}".format(Charm))
print("Vomma      : {:.10f}".format(Vomma))
print("Veta       : {:.10f}".format(Veta))

print(">Third Order-----------------------<")
print("Speed      : {:.10f}".format(Speed))
print("Zomma      : {:.10f}".format(Zomma))
print("Color      : {:.10f}".format(Color))
print("Ultima     : {:.10f}".format(Ultima))

print(">Non Greek-------------------------<")
print("Fair Value : {:.10f}".format(Fair_Value))
print("POP        : {:.10f}".format(POP))
print("Breakeven %: {:.10f}".format(Breakeven_percent))
print("Breakeven $: {:.2f}".format(Breakeven_dollars))
print("Breakeven  : {:.2f}".format(Breakeven_price))
