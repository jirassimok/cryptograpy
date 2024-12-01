import math
from random import randrange, sample as randsample

from .factors import factorize
from .fastexp import fastexp
from .prime import is_prime
from .util import is_verbose, printer, supstr, Verbosity

# TODO: Split this functions out into appropriate modules or
#       at least rename this one.


## Primitive Root search algorithm

def primitive_root(p, *,
                   nocheck=False,
                   smallest=True,
                   base_tries=5,
                   try_first=(),
                   verbose_fastexp=False):
    """Find a primitive root of prime p.

    Always checks the numbers in try_first before any others, then performs
    a search determined by the other keyword arguments.

    If smallest is True (the default), the search is performed in ascending
    order. Otherwise, try_first random values are tested to try to find a
    random primitive root before falling back to an exhaustive search.

    For small p, the fallback exhaustive search is performed in random order.
    For large p, it is performed in ascending order.

    Parameters
    ----------
    p : int
        A prime number.

    Keyword Parameters
    ------------------
    nocheck : bool, default False
        If true, don't check that p has primitive roots before searching.
    try_first : iterable of int
        If given, test these numbers first.
    smallest : bool, default True
        If true, search for primitive roots in ascending order, ignoring
        all of the remaining arguments.
    base_tries : int, default 5
        Test this many random numbers before giving up on the random search.
    verbose_fastexp : bool or None, default False
        Passed to the verbose parameter of fastexp.
    """
    if p == 2:
        return 1
    elif p == 4:
        return 3

    if not nocheck:
        # TODO: Use a faster prime test here.
        # Reject all the numbers without primitive roots
        fs = factorize(p)
        if not (len(fs) == 1
                # above: prime^k, below: 2*prime^k
                or (len(fs) == 2 and fs[2] == 1)):
            raise ValueError(f'{p} has no primitive roots')

        if not is_prime(p):
            raise NotImplementedError(
                "primitive_root does not currently support non-primes")

    phi = p - 1
    factors = factorize(phi).keys()
    exponents = [phi // q for q in factors]
    def is_root(b):
        return all(fastexp(b, x, p, verbose=verbose_fastexp) != 1
                   for x in exponents)
        # for x in exponents:
        #     if fastexp(b, x, p) == 1:
        #         return False
        # else:
        #     return True

    ## Search below

    for b in try_first: # provided guesses
        if is_root(b):
            return b

    if smallest:
        for b in range(2, p):
            if is_root(b):
                return b
        assert False, f"unreachable: primitive roots mod {p} should exist"

    for _ in range(base_tries): # random guesses
        b = randrange(2, p)
        if is_root(b):
            return b

    # Exhaustive search, in random order for very small p, ascending order
    # otherwise.
    bs = range(2, p)
    if p <= 10_000:
        # Shuffle the items, but now it
        # actually takes space.
        bs = randsample(bs, len(bs))
    for b in bs:
        if is_root(b):
            return b
    assert False, f"unreachable: primitive roots mod {p} should exist"


# TODO: Use faster factorization
def is_primitive_root(b, p, *, factors=None, verbose_fastexp=False):
    """Check if b is a primitive root of prime p.

    Keyword Parameters
    ------------------
    factors : iterable of int, optional
        An iterable of the unique prime factors of p - 1. If not given,
        these will be calculated.
    verbose_fastexp : bool or None, default False
        Passed to the verbose parameter of fastexp.
    """
    # TODO: Can I just substitute totient() and use this for non-primes?
    #       If not, I could check for primitive-rootable-numbers.
    if not is_prime(p):
        raise ValueError('p must be prime')
    if factors is None:
        phi = p - 1
        factors = factorize(phi).keys()
    else:
        phi = p - 1
    exponents = [phi // q for q in factors]
    return all(fastexp(b, x, p, verbose=verbose_fastexp) != 1
               for x in exponents)


## Baby-step Giant-step algorithm

def bsgs_log(x: int, base: int, modulus: int, *,
             verbose: Verbosity = None) -> int:
    """Compute the discrete log of x with the given base and modulus.

    Note that the modulus must be prime.

    Keyword Parameters
    ------------------
    verbose : bool, optional
        If false, print nothing. If true, or if not given and util.VERBOSE
        is true, print the steps of the algorithm in a table-like format.
    """
    print = printer(is_verbose(verbose))
    if is_verbose(verbose):
        def printrow():
            print(f'{k:>{_width}}\t{giant_acc:>3}\t{baby_acc:>3}')
    else:
        def printrow():
            pass

    order = modulus - 1 # "n"
    bound = math.ceil(math.sqrt(order)) # "m"; could use 1+math.isqrt(order-1)

    # TODO: Check preconditions
    # - base and modulus coprime (i.e. primitive root)
    # - modulus > 1
    # - base, x > 1?

    _width = len(str(bound))
    # sup_minus_m = '\N{superscript minus}\N{modifier letter small m}'
    # sup_i = '\N{superscript latin small letter i}',
    # sup_j = '\N{modifier letter small j}'

    base_inv = fastexp(base, order - 1, modulus, verbose=False)
    giant_step_factor = fastexp(base_inv, bound, modulus, verbose=False)

    print(f'b⁻¹ = bᵠ⁽ⁿ⁾⁻¹ = {base}^({order} - 1) mod {modulus}'
          f' ≡ {base_inv}')
    print(f'b⁻ᵐ = (b⁻¹)ᵐ = {base_inv}^{bound} mod {modulus}'
          f' ≡ {giant_step_factor}')
    print()

    giant_acc = x
    baby_acc = 1

    giant_steps = {x: 0}
    baby_steps = {1: 0}

    # a = x
    # b = base
    # n = order
    # c = giant_step_factor

    print('i or j', 'a(b⁻ᵐ)ⁱ', 'bʲ', f'(mod {modulus})', sep='\t')
    k = 0
    printrow()
    for k in range(1, bound):
        giant_acc = (giant_acc * giant_step_factor) % modulus
        baby_acc = (baby_acc * base) % modulus
        printrow()
        giant_steps[giant_acc] = k
        baby_steps[baby_acc] = k
        if giant_acc in baby_steps:
            i, j = k, baby_steps[giant_acc]
            print(f'found a(b⁻ᵐ){supstr(i)} = {giant_acc} = b{supstr(j)}')
            break
        elif baby_acc in giant_steps:
            i, j = giant_steps[baby_acc], k
            print(f'found b{supstr(j)} = {baby_acc} = a(b⁻ᵐ){supstr(i)}')
            break
    else:
        # no match found
        raise ValueError('no match found')

    exp = i * bound + j
    print(f'i×m + j = {i}×{bound} + {j} = {exp} ≡ {exp % order} mod {order}')
    return exp % order


def discrete_log(x: int, base: int, modulus: int) -> int:
    """Calculate a discrete logarithm using the baby-step giant-step algorithm.

    See bsgs_log for details.
    """
    return bsgs_log(x, base, modulus, verbose=False)
