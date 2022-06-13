# Linearization of NN Surrogate Model

We compute linearization of the NN surrigate f(x) = y, to understand sensitivity to different inputs.

Linearization of f(x`) = Ax` + b, can be computed by computing the gradiant grad f(x) around x`. Since f(x) is vector valued we compute a jacobian: A = J_x(f(x)). 

To visualize how much each input perturbs every output, we normalize each row of J. The following figure breaks up J by different output and input groups.

## CAM4
[![](jacobian.png)](jacobian.html)