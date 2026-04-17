# ruff: noqa: D103
import time

import numpy as np


def brute_force_creation(ds):
    # Since this is a no-op and takes < 1 us, we need to delay it or the timing
    # code is utterly dominated by the time to reset the data (which scales as
    # n*log(n))
    # We'll make it .2 ms so that we only need to run 1000 loops
    time.sleep(0.0002)
    return ds.positions


def brute_force_search(
    positions, *, centers, radii, particle_numbers: list[int], **kwargs
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        mask = np.sum((positions - c) ** 2, axis=1) <= r**2
        number = np.sum(mask)
        if particle_numbers and number != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and got {number}.
                """
            )
