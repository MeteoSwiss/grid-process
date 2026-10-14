# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module for optimization of computations
"""

import logging
from functools import wraps
from typing import Any, TypeVar, cast
from collections.abc import Callable, Hashable

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def memoize(maxsize: int = 10) -> Callable[[F], F]:
    """
    LRU cache decorator implementation for any arbitrary input since the
    caching is purely based on the optional keyword argument ``hkey``.
    """

    def real_memoize(func: F) -> F:
        cache: dict[Hashable, Any] = {}
        hkeys = []

        @wraps(func)
        def _memoize(*args: Any, **kwargs: Any) -> Any:
            hkey = kwargs.pop("hkey", None)
            if hkey in cache:
                logger.debug("@memoize: %s cached!", func.__name__)
                return cache[hkey]
            logger.debug("@memoize: %s computing ...", func.__name__)
            result = func(*args, **kwargs)
            if hkey is not None:
                cache[hkey] = result
                hkeys.append(hkey)
                if len(hkeys) > maxsize:
                    cache.pop(hkeys.pop(0))
                logger.debug("@memoize: %s cache size is %s}", func.__name__, len(cache))

            return result

        return cast(F, _memoize)

    return real_memoize
