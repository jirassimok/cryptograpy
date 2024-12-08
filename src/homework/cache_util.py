from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator


class Cache(defaultdict[int, int]):
    """A defaultdict that passes its missing keys to its default factory.
    """
    def __init__(self, default_factory: Callable[[int], int]):
        super().__init__(default_factory)  # type: ignore[arg-type]

    def __missing__(self, key):
        return self.default_factory(key)


class CachingIterable[T](Iterable[T]):
    """An iterable that caches its values the first time they are seen.
    """
    def __init__(self, base: Iterable[T]):
        self.base = iter(base)
        self.cache: list[T] = []

    def __iter__(self) -> Iterator[T]:
        cache = self.cache
        it = self.base
        i = -1
        while True:
            i += 1
            if i >= len(cache):
                try:
                    new = next(it)
                except StopIteration:
                    return
                cache.append(new)
                yield new
            else:
                yield cache[i]
