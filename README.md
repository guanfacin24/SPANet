## SPANet (RWTH + DESY fork)
This is a fork of Alexander Shmakov's SPANet repository with various modifications that we need for our physics goals (and conveniance).
+ SPANet base version: v2.2.0
+ Fork verson: v0.1.1

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

### Improved progress bar logging
In SPANet's vanilla configuration, the progress bar logging only contains the average jet accuracy of the validation dataset and whether the current network is within the top three best performing network iterations. Especially when training a combined network (assignment + classification for example), this logging method is suboptimal, as it would be much more beneficial to be able to track each part of the combined network **separately**. Moreover, even when training on assignment alone, the Higgs accuracy would be of much more value than the average jet accuracy (even though they are of course stronly correlated).

In this fork, we are able to specify which metrics we would like to track in the progress bar and also which metric should be used for creating checkpoints. This can be done quite easily in the options file by adding:
```
{
  ...

  "tracking_metrics": ["EVENT/H_accuracy", "CLASSIFICATION/Signal_accuuracy"],
  "central_metric": ["CLASSIFICATION/Signal_accuracy", "max"]
}
```
In this case, our Higgs is called "H" in the event file and our classification "Signal". By default, the top three best performing networks with regards to the average jet accuracy are saved as checkpoints. But by using the `central_metrics` option, the metrics with which these checkpoints are created can be changed to something else, like the classification accuracy for example. The second entry of this list (`"max"`) determines whether this metrics shall be maximized or minimized.
> **Note:** No matter what central metric is given, the training will still minimize the loss, which is often a combination of different types of losses (assignment, detection, classification, etc...) with its mixture being specified in the options file. It would also make sense to be able to track the validation losses of these individual loss types. This is however not implemented yet. The best option is to use **tensorboard** to retroactively investigate all different loss curves.
