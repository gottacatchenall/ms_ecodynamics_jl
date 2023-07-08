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

Start with simple case, only two patches, $i$ and $j$.

What is the probability that they are more than $r$ apart?

$X_i \sim \text{U}(0,1)$, $X_j \sim \text{U}(0,1)$.

What is the distribution of $X_i - X_j$? We can express this using a convolution
of $X_i$ and $X_j$. 

We define $Z = X_i - X_j$.

$$P(Z=z) = \int_0^z P(X=x)P(Y=z+x) dx$$