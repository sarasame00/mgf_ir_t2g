import numpy as np

class OhMatrix:
    """
    Represents a matrix of the form M = a·I + b·V, where V is a fixed 3×3 matrix structure.
    This form appears naturally in the t₂g basis with spin-orbit or Jahn-Teller interactions,
    and allows for algebraic operations to be done efficiently using scalar coefficients a and b.

    Supports both scalar and array-valued a, b. Enables broadcasting, IR basis evaluation,
    and custom linear algebra for physical simulations.
    """

    def __init__(self, a, b):
        # Case 1: scalar inputs (e.g. single value per matrix)
        if isinstance(a, (int, float, complex)) and isinstance(b, (int, float, complex)):
            self.__a = a
            self.__b = b
            self.__is_array = False
            return

        # Case 2: array-valued a, b (e.g. sampled over frequency, space, etc.)
        try:
            _a = np.array(a)
            _b = np.array(b)
        except:
            raise TypeError("Failed to cast inputs to arrays")

        # Check that element types are numeric
        if (type(_a.flatten()[0].item()) in [int, float, complex] and
            type(_b.flatten()[0].item()) in [int, float, complex]):
            try:
                # Ensure shape compatibility
                _a = _a * np.ones_like(_b)
                _b = _b * np.ones_like(_a)
            except:
                raise ValueError("Incompatible array shapes")
            self.__a = _a
            self.__b = _b
            self.__is_array = True
        else:
            raise TypeError("Non-numeric input types not supported")

    # ----------------------
    # Properties and Metadata
    # ----------------------

    @property
    def a(self): return self.__a

    @property
    def b(self): return self.__b

    @property
    def shape(self): return np.asarray(self.a).shape

    @property
    def size(self): return np.asarray(self.a).size

    @property
    def ndim(self): return np.asarray(self.a).ndim

    @property
    def real(self): return ohmatrix(self.a.real, self.b.real)

    @property
    def imag(self): return ohmatrix(self.a.imag, self.b.imag)

    @property
    def eigvals(self):
        # Eigenvalues of the 3×3 matrix a·I + b·V using V² = 2I + V
        return self.a + 2*self.b, self.a - self.b

    @property
    def trace(self):
        # Tr[M] = 3a + 3a = 6a (since I and V are traceless in opposite ways)
        return 6 * self.a

    # ----------------------
    # Linear Algebra
    # ----------------------

    def inv(self):
        # Efficient inverse using known structure of V
        denom = self.a*self.a - 2*self.b*self.b + self.a*self.b
        c = (self.a + self.b) / denom
        d = -self.b / denom
        return ohmatrix(c, d)

    def reshape(self, newsh):
        # Reshape both a and b arrays
        return ohmatrix(np.asarray(self.a).reshape(newsh),
                        np.asarray(self.b).reshape(newsh))

    def __getitem__(self, key):
        return ohmatrix(self.a[key], self.b[key])

    def __setitem__(self, key, item):
        if not isinstance(item, OhMatrix):
            raise TypeError("Only OhMatrix items can be assigned")
        self.__a[key] = item.a
        self.__b[key] = item.b

    # ----------------------
    # Functional Mappings
    # ----------------------

    def __apply_func(self, func):
        # General function mapping via eigenvalues
        c = (func(self.a + 2*self.b) + 2*func(self.a - self.b)) / 3
        d = (func(self.a + 2*self.b) - func(self.a - self.b)) / 3
        return ohmatrix(c, d)

    def exp(self): return self.__apply_func(np.exp)
    def cos(self): return self.__apply_func(np.cos)
    def sin(self): return self.__apply_func(np.sin)
    def tan(self): return self.__apply_func(np.tan)
    def log(self): return self.__apply_func(np.log)
    def sqrt(self): return self.__apply_func(np.sqrt)
    def cbrt(self): return self.__apply_func(np.cbrt)

    # ----------------------
    # Arithmetic Operators
    # ----------------------

    def __pos__(self): return OhMatrix(self.a, self.b)
    def __neg__(self): return OhMatrix(-self.a, -self.b)

    def __add__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a + other.a, self.b + other.b)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return ohmatrix(self.a + other, self.b)
        raise TypeError

    def __radd__(self, other): return self.__add__(other)
    def __iadd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a - other.a, self.b - other.b)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return ohmatrix(self.a - other, self.b)
        raise TypeError

    def __rsub__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(other.a - self.a, other.b - self.b)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return ohmatrix(other - self.a, -self.b)
        raise TypeError

    def __isub__(self, other): return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, OhMatrix):
            # Matrix multiplication rule derived from V² = 2I + V
            a = self.a * other.a + 2 * self.b * other.b
            b = self.b * other.a + (self.a + self.b) * other.b
            return ohmatrix(a, b)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return ohmatrix(self.a * other, self.b * other)
        raise TypeError

    def __rmul__(self, other): return self.__mul__(other)
    def __imul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, OhMatrix):
            return self * other**-1
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return ohmatrix(self.a / other, self.b / other)
        raise TypeError

    def __rtruediv__(self, other):
        return self.inv() * other

    def __itruediv__(self, other): return self.__truediv__(other)

    def __pow__(self, other):
        if isinstance(other, int):
            if other == 1:
                return self
            elif other == 0:
                return OhMatrix(1, 0)
            elif other > 1:
                return self**(other - 1) * self
            elif other < 0:
                return (self**-other).inv()
        elif isinstance(other, (float, complex, np.ndarray)):
            return self.__apply_func(lambda x: x**other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Numpy ufunc override for compatibility
        if ufunc.nin == 1:
            return self.__apply_func(ufunc)
        elif ufunc in [np.add, np.subtract]:
            arr, ohm = inputs
            a = ufunc(arr, ohm.a, **kwargs)
            return ohmatrix(a, ohm.b)
        elif ufunc in [np.multiply, np.divide]:
            arr, ohm = inputs
            if ufunc == np.divide:
                ohm = ohm.inv()
            return ohmatrix(arr * ohm.a, arr * ohm.b)
        raise NotImplementedError

    def __repr__(self):
        return f"OhMatrix({self.a}, {self.b})"

# ----------------------
# Constructors
# ----------------------

def ohmatrix(a, b):
    """
    Public constructor for OhMatrix.
    Ensures standard usage style: ohmatrix(a, b)
    """
    return OhMatrix(a, b)

def ohzeros(shape):
    """
    Creates an OhMatrix with zero a and b components.
    """
    return OhMatrix(np.zeros(shape), np.zeros(shape))

def ohrandom(shape):
    """
    Creates a random OhMatrix with complex-valued entries (magnitude + random phase).
    """
    ar = np.random.random(shape)
    ap = 2 * np.pi * np.random.random(shape)
    br = np.random.random(shape)
    bp = 2 * np.pi * np.random.random(shape)
    a = ar * np.exp(1j * ap)
    b = br * np.exp(1j * bp)
    return OhMatrix(a, b)

# ----------------------
# Utility operations
# ----------------------

def ohsum(M, **kwargs):
    """
    Sum an OhMatrix over given axes (calls np.sum on .a and .b separately).
    """
    return ohmatrix(np.sum(M.a, **kwargs), np.sum(M.b, **kwargs))

def ohcopy(M, **kwargs):
    """
    Deep copy of an OhMatrix (copies both a and b fields).
    """
    return ohmatrix(np.copy(M.a), np.copy(M.b))
