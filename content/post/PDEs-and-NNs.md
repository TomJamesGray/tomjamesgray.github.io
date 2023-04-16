---
title: "Neural Networks and Solving PDEs"
date: 2023-04-16T11:21:50+01:00
draft: true
math: true
---

In recent years machine learning has become immensely popular with an incredible amount of research being done on the area. The term, in my opinion, is quite a loosely defined one but in many cases boils down to a gradient descent approach to minimise a cost function. 
<!--more-->


## Neural Network Structure



## Cost Functions

In order for a machine to *learn* what the solution is to a problem, we need to give it a way of telling when a given guess at the solution or ansatz is good. This is typically done with a cost function which measures how far away some ansatz is from the true solution. The goal of the neural network is to minimise this function.

<!-- ![Example cost function](grad_descent.jpg) -->

For instance say our cost function is $C(p_1, p_2) = 3 p_1^2 + 2 \sin(p_2)$ where $p_1, p_2$ are the parameters of our neural network. In this case we can tell the minima of $C(p_1, p_2)$ is $1$ as 

$$
3p_1^2 \geq 0 \quad \text{for all} \quad p_1 \\\
2\sin(p_2) \geq -2 \quad \text{for all} \quad p_2
$$

We can also see that we achieve this minima whenever $p_1 = 0$ and 
$$
p_2 = -\frac{\pi}{2} + 2n\pi
$$
where $n$ is some integer.

Typically the cost function for neural networks will be much more complex and can, in a sense, be treated as a black box where we give it the parameters of our network and it tells us whether our ansatz is good or terrible. Therefore to minimise some arbitrary cost function for a neural network we need to be a little bit more creative. This is where gradient descent comes into play.

### Gradient Descent

A common way of thinking about gradient descent is imagining the function you are minimising is a hill and it is a foggy day. In order to find the bottom of that hill you would take steps in the downard direction. While this might not necessarily be the fastest route it will get you to a bottom of the hill.

Returning to our example of minimising our cost function $C(p_1, p_2) = 3 p_1^2 + 2 \sin(p_2)$, imagine we have randomly initialised our parameters as $(p_1, p_2)$ = $(1,1)$. This gives us $C(1,1) \approx 4.68$. 

Now in order to determine what the downward direction is we need to compute the derivative of $C(p_1,p_2)$ when $(p_1,p_2) = (1,1)$. This can be written as
$$
\nabla C(p_1,p_2) = 
\begin{pmatrix}
    \frac{\partial C}{\partial p_1} \\\
    \frac{\partial C}{\partial p_2}
\end{pmatrix} = 
\begin{pmatrix}
    6p_1 \\\
    2 \cos(p_2)
\end{pmatrix}
$$
Evaluating this at our point $(1,1)$ gives us $\begin{pmatrix}6 \\ 1.1\end{pmatrix}$. This tells us how steep $C(p_1,p_2)$ is if we just care about $p_1$ or $p_2$.

Returning to the hill analogy if we know what the steepest direction is, in order to find the minima we shall go the opposite direction to that. Therefore we generate new guesses of $p_1,p_2$ which we could call $p^{[1]}_1$, $p^{[1]}_2$ as
$$
\begin{pmatrix}
    p^{[1]}_1 \\\
    p^{[1]}_2
\end{pmatrix} = 
\begin{pmatrix}
    p_1 \\\
    p_2
\end{pmatrix} - \eta
\nabla C(p_1,p_2)
$$
where $\eta$ is some parameter which is commonly called the *learning rate*. This just determines how big of a step we should take. 

For instance we could take $\eta$ as $0.1$ which would give us $\begin{pmatrix}
    p^{[1]}_1 \\\
    p^{[1]}_2
\end{pmatrix} = \begin{pmatrix} 0.4 \\\ 0.89 \end{pmatrix}$. Evaluating $C$ with these new parameters gives us a value of $2.04$ which is much smaller than what we got initially which was $4.68$. This process can be repeated many times and each time we *should* get closer to finding a minima of $C$

## Cost function for PDEs

PDEs are commonly written in a form referred to as the *strong form*, for instance $u'' = g$ where $u$ is the function we want to solve for, and $g$ is some function we know. However there is another form which is reffered to as the *variational form*. This form reframes solving some derivative (or combinations of) of $u$ being equal to some function as finding the function which minimises some other function. This is very useful for us as this minimisation idea fits well with neural networks.

We shall consider Poisson equations equations in 2 dimensions, that is to say the equations will be of the form

$$
-\Delta u(x,y) = 
-\frac{\partial^2 u}{\partial x^2} - \frac{\partial u}{\partial y^2}
= f(x,y)
$$

and we shall solve over some arbitary domain $\Omega \subseteq \mathbb{R}^2$. We can also enforce homogeneous Dirichlet boundary conditions to make some of the derivations nicer, that is to say $u(x,y) = 0$ on the boundary of $\Omega$, denoted $\partial \Omega$

## Weak form

To get to the variational form we first need to get to the *weak form* of the equation. To do this we shall multiply the above equation by some other function $v(x,y)$ which is defined over $\Omega$ and integrate over the domain $\Omega$. This gives us 

$$
-\int_{\Omega} (\Delta u)v \mathrm{d} \mathbf{x} = 
\int_{\Omega} fv \mathrm{d} \mathbf{x}
$$

Strictly speaking $v$ must satisfy some regularity conditions, rigorously speaking it must be in the Sobolev space $W^{1,2}(\Omega)$. In effect this means $v$ and its first partial derivatives must be suitably continous. Furthermore we can enforce $v$ satisfies the boundary conditions of being zero on the boundary of $\Omega$, i.e. $v = 0$ on $\partial \Omega$

We can further simplify the above equation using Green's first identity which gives us 
$$
\begin{aligned}
-\int_{\Omega} (\Delta u)v \mathrm{d} \mathbb{x} 
&= \int_{\Omega} {\nabla u \cdot \nabla v} \mathrm{d}\mathbf{x}
    - \int_{\partial \Omega} v \nabla u \cdot \mathbf{n} \mathrm{d}s\\\
&= \int_{\Omega} f v \mathrm{d}\mathbf{x}
\end{aligned}
$$
where $\mathbf{n}$ is the outward facing normal derivative. As $v=0$ on $\partial \Omega$ we get
$$
\int_{\Omega} {\nabla u \cdot \nabla v} \mathrm{d}\mathbf{x}
= \int_{\Omega} fv \mathrm{d}\mathbf{x}
$$

## Variational form

The variational form of the above weak form can be given as finding the function $u$ which minimizes
$$
F(v) = \frac{1}{2}\int_{\Omega} \nabla v \cdot \nabla v - fv \mathrm{d}x
$$
over all admissable functions. 

To show this we must show that if $u$ satisfies the weak form it satisfies the variational form and vice versa. The former can be done by considering $F(u+w)$ where $w$ is some arbitrary function and then showing that $F(u+w) > F(u)$. The latter can be shown by considering $F(u+\epsilon v)$ where $\epsilon \in \mathbb{R}$. By treating $v$ as some known function we can write $F(u+\epsilon v)$ as a function in $\epsilon$ and then show we achieve a minima at $\epsilon=0$.

Now we have a way of quantifying how good a given function is at approximating a PDE we can see how the application of neural networks to PDEs seems natural.

## Existing research

