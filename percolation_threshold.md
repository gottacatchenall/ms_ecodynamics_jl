Major questions:

1. Each of $n$ points is selected uniformly in the unit square, i.e. 
for a point $a_i = (x_i, y_i)$, both $x_i$ and $y_i \sim \text{Uniform}(0,1)$. 

2. For a given distance $r$, what is the probability that for a given point $a_i$,
no other point $a_j$ is within $r$ of it?

3. Then, what is the probability that there exists one or more points $a_i$ such
that no other point $a_j$ is within $r$ of $a_i \ \forall j \neq i$? 

4. What does that probability approach for a given $r$ and $n \to \infty$?

5. And finally, what is the smallest value of $r$ where $n \to \infty$?


---

# Part One: probability single pair is more than $r$ apart

Start with simple case, only two patches, $i$ and $j$.
What is the probability that they are more than $r$ apart?

$X \sim \text{U}(0,1)$, $X \sim \text{U}(0,1)$.

What is the distribution of $X - Y$? We can express this using a convolution of
$X$ and $Y$.

We define $Z = | X - Y |$. The pdf of $Z$, which is the pdf of the distribution we
are interested in, is given by  


Eventually, $$ P(z) = 2 - \frac{z}{2} $$


We are interested in a new random variable $D = \sqrt{(X_1 - Y_1)^2 + (X_2 -
Y_2)^2}$ where $X_i$ and $Y_i$ and i.i.d. draws from $\text{Uniform}(0,1)$. 

We are interested in the probably 

$$P(D \leq r)$$



$$P(\text{Any isloated}) = 1 - P(\text{None isolated})$$