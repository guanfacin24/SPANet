## SPANet (RWTH + DESY fork)
This is a fork of Alexander Shmakov's SPANet repository with various modifications that we need for our physics goals (and conveniance).
+ SPANet base version: v2.2.0
+ Fork verson: v0.0.1

Here's a (likely incomplete) list of the new features we have implemented:
### Custom event weights
This new feature allows you to assign custom event-wise weights to the loss function, such that 

$$ \mathcal{L} = \sum_{i} w_i \cdot \mathcal{L}_i$$

with $\mathcal{L}$ being the total loss, $\mathcal{L}_i$ the loss associated with event $i$ and $w_i$ the weight assigned to event $i$. The weights need to be stored in the training file under `WEIGHTS/EVENT/<weight-name>` and must naturally be of same length as all other arrays. The event file must be modified as well to include:
```
WEIGHTS:
  EVENT:
    - <weight-name>
    - <...>
```
Note that it is possible to include multiple weights as well, which will then be multiplied before training.
