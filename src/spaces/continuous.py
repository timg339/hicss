from typing import SupportsFloat, Any, Union, Optional, Sequence

import gymnasium as gym
import numpy as np


class ContinuousActionSpace(gym.spaces.Box):
    """ Parent class for variable continuous action spaces """

    def __init__(self, low: SupportsFloat, high: SupportsFloat,
                 shape: Sequence[int] | None = None,
                 dtype: Union[type[np.floating[Any]], type[np.integer[Any]]] = np.float32,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().__init__(low, high, shape=shape, dtype=dtype, seed=seed)

    def reset(self):
        """ Resets the action space to its initial state """

        raise NotImplementedError
