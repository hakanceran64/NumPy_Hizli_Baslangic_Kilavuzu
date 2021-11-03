![NumPy Photo](./NumPy_Python_Cover.jpg)

---

# Numpy Hızlı Başlangıç Kılavuzu

[Numpy Quick Start](https://numpy.org/doc/stable/user/quickstart.html)

## İçindekiler
  - [Ön Koşullar](#prerequisites)
  - [Temeller](#basics)
      - [Array Oluşturma](#array_creation)
      - [Array'lerin Yazdırılması](#printing_array)
      - [Temel İşlemler](#basic_operations)
      - [Evrensel Fonksiyonlar](#universal_functions)
      - [İndeksleme, Dilimleme ve Yineleme](#indexing_slicing_and_iterating)

---

## <a id="prerequisites" /> Önkoşullar

- Birazcık *Python* bilgisine sahip olmanız gerek.
- Bazı örnekleri çalıştırmak için *matplotlib* kütüphanesini kurmak zorunda kalabilirsiniz.

### Öğrenme Profili

- NumPy Array'lerine hızlı bir bakış atacaksınız.
- Array'lerin nasıl tanımlandığını ve manipüle edildiğini öğreneceksiniz.
- For döngüsü kullanmadan genel işlevleri uygulayabileceksiniz.
- Array'lerde axis ve shape özelliklerine hakim olacaksınız.

### Hedefler

- Bu makaleyi okuduktan sonra şunları yapabiliyor olmanız gerekir:
  - NumPy'de dimension nedir ve 1, 2 veya daha fazla dimension'a sahip Array'ler arasındaki farkı anlamış olmanız gerekmektedir.
  - Temel işlemlerin for döngüsü kullanılmadan NumPy Array'lerine nasıl uygulandığını öğrenmiş olmanız gerekmektedir.
  - NumPy Array'ler için axis ve shape özelliklerini anlamış olmanız gerekmektedir.

---

## <a id="basics" /> Temeller

3 boyutlu uzayda [1, 3, 5] olarak tanımlanan elemanlar dizisi aslında bir noktanın koordinatlarını temsil eder ve bildiğiniz gibi bir noktanın tek bir ekseni vardır. Bu noktanın kooardinatlarını ifade eden [1, 3, 5] elemanlar dizisinin 3 adet elemanı olduğunu söyleyebiliriz. Şimdi aşağıda ifade edilen elemanlar dizisine bakalım.

~~~Python
[[1., 3., 5.],
 [4., 6., 8.]]
~~~

Burada 2 eksen vardır. XY kooardinat sistemi gibi düşünebilirsiniz. İlk eksenin uzunluğu 2 ve ikinci eksenin uzunluğu ise 3'tür. İlk eksen olarak ifade ettiğimiz eksen (Kooardinat sisteminde Y ekseni) aslında NumPy'de *axis = 1* olarak ifade edilmektedir. İkinci eksen olarak ifade ettiğimiz eksen ise (Kooardinat sisteminde X ekseni) NumPy'da *axis = 0* olarak ifade edilmektedir.

NumPy'ın Array sınıfına aslında *ndarray* denilmektedir. NumPy Array'lerinin Python'daki Array'ler ile aynı şey olmadığını unutmayın. Şimdi NumPy Array'lerinin (ndarray) önemli özelliklerinden bazılarına bakalım:

- **ndarray.ndim** (Array'in axis sayısını(dimensions) döndürür.)
- **ndarray.shape** (Satır(rows) ve sütun(columns) sayısını döndürür.)
- **ndarray.size** (Array'ın toplam eleman sayısını verir. Satır ve sütun çarpımına eşittir.)
- **ndarray.dtype** (Array'deki elemanların türünü açıklar. NumPy'ın kendi türlerini tanımladığını unutmayın. numpy.int32, numpy.int16 ve numpy.float64 gibi...)
- **ndarray.itemsize** (Array'ın her bir elemanının bayt cinsinden boyutu. Aslında bu ifade *ndarray.dtype.itemsize* şeklinde de kullanılabilir.)
- **ndarray.data** (Array'ın elemanlarının içeren bellek adresini tanımlar. Aslında buna çok ihtiyacımız yok. Bakınız: *Indexing Facilities*)

### Bir Örnek

~~~Python
>>> import numpy as np

>>> a = np.arange(20).reshape(4, 5)

>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])

>>> a.shape
(4, 5)

>>> a.ndim
2

>>> a.dtype.name
'int64'

>>> a.itemsize
8

>>> a.size
20

>>> type(a)
<class 'numpy.ndarray'>

~~~

---

### <a id="array_creation" /> Array Oluşturma

NumPy'da Array oluşturmanın birden çok yöntemi vardır. Örneğin dizi özelliklerini kullanarak Python listesinden bir NumPy Array oluşturabilirsiniz. Oluşturduğunuz NumPy Array'ın türü ise elemanları doğrultusunda değişkenlik göstermektedir. Çok sık karşılaşılan bir hata Array tanımlamalarında [] ifadesini unutmayınız.

~~~Python

>>> import numpy as np

>>> a = np.array([1, 2, 3])

>>> a
array([1, 2, 3])

>>> a.dtype
dtype('int64')

>>> b = np.array([1.2, 2.2, 3.2])

>>> b.dtype
dtype('float64')

>>> c = np.array(["1", "2", "3"])

>>> c.dtype
dtype('<U3')

~~~

Array'ler dizi özelliklerinden faydalanarak 2 boyutlu, 3 boyutlu veya daha fazla NumPy Array'ler tanımlanmasına imkan tanımaktadır.

~~~Python

>>> import numpy as np

>>> a = np.array([[1, 2, 3], [4, 5, 6]])

>>> a
array([[1, 2, 3],
       [4, 5, 6]])

>>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

>>> b
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

~~~

Array'lerin türünü oluşturma sırasında belirtilebiliriz:

~~~Python

>>> import numpy as np

>>> a = np.array([[1, 2], [3, 4]], dtype=complex)
>>> a
array([[1.+0.j, 2.+0.j],
       [4.+0.j, 4.+0.j]])

~~~

Peki ya oluşturacağımız Array'ın ögelerini ilk başta bilmiyorsak? NumPy bize bu durumda boyutunu bildiğimiz fakat ögelerini bilmediğimiz Array'ler için yer tutabilmemiz için bir çok imkan sağlamaktadır. Bu imkanlar sayesinde sonradan büyüyen Array'lerden kurtulmuş oluruz.

~~~Python

>>> import numpy as np

>>> np.zeros([3, 4])
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])

>>> np.ones([2, 3, 4], dtype=np.int16)
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],

       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]], dtype=int16)

>>> np.empty([2, 3])
array([[4.65554319e-310, 0.00000000e+000, 0.00000000e+000],
       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000]])

~~~

Sayı dizileri oluşturmak istediğimizde NumPy bize Python'daki *range* yapısına benzeyen *arange* yapısını kullanarak Array'ler döndürür.

~~~Python

>>> import numpy as np

>>> np.arange(10, 30, 5)
array([10, 15, 20, 25])

>>> np.arange(0, 2, 0.3)
array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])

~~~

*arange* ile Array oluşturduğumuzda elde edilen öğelerin sayısını tahmin edemiyoruz. Bu da bize çok büyük uzunluklarda öğeler döndürme ihtimalinin olduğunu gösteriyor. Bu nedenle *arange* yerine istediğimiz sayıda öğe döndüren *linspace* yapısını kullanmak daha iyi bir tercih olabilir. Aşağıda bu konu ile alakalı bir örnek göreceksiniz.

~~~Python

>>> from numpy import pi

>>> np.linspace(0, 2, 9)

>>> x = np.linspace(0, 2 * pi, 100)
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])

>>> f = np.sin(x)
array([ 0.00000000e+00,  6.34239197e-02,  1.26592454e-01,  1.89251244e-01,
        2.51147987e-01,  3.12033446e-01,  3.71662456e-01,  4.29794912e-01,
        4.86196736e-01,  5.40640817e-01,  5.92907929e-01,  6.42787610e-01,
        6.90079011e-01,  7.34591709e-01,  7.76146464e-01,  8.14575952e-01,
        8.49725430e-01,  8.81453363e-01,  9.09631995e-01,  9.34147860e-01,
        9.54902241e-01,  9.71811568e-01,  9.84807753e-01,  9.93838464e-01,
        9.98867339e-01,  9.99874128e-01,  9.96854776e-01,  9.89821442e-01,
        9.78802446e-01,  9.63842159e-01,  9.45000819e-01,  9.22354294e-01,
        8.95993774e-01,  8.66025404e-01,  8.32569855e-01,  7.95761841e-01,
        7.55749574e-01,  7.12694171e-01,  6.66769001e-01,  6.18158986e-01,
        5.67059864e-01,  5.13677392e-01,  4.58226522e-01,  4.00930535e-01,
        3.42020143e-01,  2.81732557e-01,  2.20310533e-01,  1.58001396e-01,
        9.50560433e-02,  3.17279335e-02, -3.17279335e-02, -9.50560433e-02,
       -1.58001396e-01, -2.20310533e-01, -2.81732557e-01, -3.42020143e-01,
       -4.00930535e-01, -4.58226522e-01, -5.13677392e-01, -5.67059864e-01,
       -6.18158986e-01, -6.66769001e-01, -7.12694171e-01, -7.55749574e-01,
       -7.95761841e-01, -8.32569855e-01, -8.66025404e-01, -8.95993774e-01,
       -9.22354294e-01, -9.45000819e-01, -9.63842159e-01, -9.78802446e-01,
       -9.89821442e-01, -9.96854776e-01, -9.99874128e-01, -9.98867339e-01,
       -9.93838464e-01, -9.84807753e-01, -9.71811568e-01, -9.54902241e-01,
       -9.34147860e-01, -9.09631995e-01, -8.81453363e-01, -8.49725430e-01,
       -8.14575952e-01, -7.76146464e-01, -7.34591709e-01, -6.90079011e-01,
       -6.42787610e-01, -5.92907929e-01, -5.40640817e-01, -4.86196736e-01,
       -4.29794912e-01, -3.71662456e-01, -3.12033446e-01, -2.51147987e-01,
       -1.89251244e-01, -1.26592454e-01, -6.34239197e-02, -2.44929360e-16])

~~~

---

### <a id="printing_array" /> Array'lerin Yazdırılması

Bir Array'ı print ettiğinizde NumPy bunu nesned list'lere benzer bir yöntemle gösterir. Bu işlemleri de aşağıdaki düzenle gerçekleştirir:

- Son eksen soldan sağa yazdırılır.
- Sondan ikinci eksen yukarıdan aşağıya doğru yazdırılır.
- Geri kalan eksenleri de yukarıdan aşağıya doğru yazdırır. Her parça kendinden sonra gelen parçadan bir satır boşlukla ayrılır.

Örneğin:

~~~Python

>>> import numpy as np

>>> a = np.arange(15) # 1D Array

>>> print(a)

[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

>>> b = np.arange(20).reshape(4, 5) # 2D Array

>>> print(b)

[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]

>>> c = np.arange(40).reshape(2, 4, 5) # 3D Array

>>> print(c)

[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]
  [15 16 17 18 19]]

 [[20 21 22 23 24]
  [25 26 27 28 29]
  [30 31 32 33 34]
  [35 36 37 38 39]]]

~~~

Eğer bir Array yazdırılamayacak kadar büyükse NumPy bu Array'ı print ederken sadece ilk ve son kısmını gösterir.

~~~Python

>>> import numpy as np

>>> a = np.arange(10000)

>>> print(a)

[   0    1    2 ... 9997 9998 9999]

>>> b = np.arange(10000).reshape(100, 100)

>>> print(b)

[[   0    1    2 ...   97   98   99]
 [ 100  101  102 ...  197  198  199]
 [ 200  201  202 ...  297  298  299]
 ...
 [9700 9701 9702 ... 9797 9798 9799]
 [9800 9801 9802 ... 9897 9898 9899]
 [9900 9901 9902 ... 9997 9998 9999]]

~~~

Yok ben her şeyi görmek istiyorum derseniz de buyurun işte bu kod parçasını çalışmanıza ekleyin.

~~~Python

>>> import numpy as np

>>> import sys

>>> np.set_printoptions(threshold=sys.maxsize)  # sys module should be imported

~~~

---

### <a id="basic_operations" /> Temel İşlemler

Array'lerde aritmatik işlemler eleman bazında uygulanır. Bunu uygularken yeni bir Array oluşturmalısınız ve sonuçları yeni oluşturduğunuz Array'e döndürmeniz gerekmektedir.

~~~Python

>>> import numpy as np

>>> a = np.array([10, 20, 30, 40, 50, 60])

>>> b = np.arange(6)

>>> b
array([0, 1, 2, 3, 4, 5])

>>> c = a - b

>>> c
array([10, 19, 28, 37, 46, 55])

>>> b**2
array([ 0,  1,  4,  9, 16, 25])

>>> 10 * np.sin(a)
array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 , -2.62374854,
       -3.04810621])

>>> a < 35
array([ True,  True,  True, False, False, False])

~~~~

Eğer matrislerde çarpma işlemi yapacaksanız * operatörü eleman bazlı işlem yaptığı için bu sizin istediğiniz cevabı elde etmenize olanak tanımaz. Hatta kare olmayan matris çarpımlarında da hata verecektir. Bunu yapmanın iki farklı yöntemi var. Birincisi *@* operatörünü kullanmak. İkincisi ise *.dot()* fonksiyonunu kullanmak. (Python sürümünüzün 3.5 ve yukarısı olduğundan emin olun)

~~~Python

>>> import numpy as np

>>> A = np.array([[1, 2],
                  [2, 0],
                  [3, 1]])

>>> B = np.array([[4, 3],
                  [1, 2]])

>>> A @ B     # matrix product
array([[ 6,  7],
       [ 8,  6],
       [13, 11]])

>>> A.dot(B)  # another matrix product
array([[ 6,  7],
       [ 8,  6],
       [13, 11]])

~~~

+=, -= *= gibi operatörler sizin yeni bir Array oluşturmadan sonucu direkt olarak varolan Array üzerine yazmanız için kullanılabilir.

~~~Python

>>> import numpy as np

>>> rg = np.random.default_rng(1)  # create instance of default random number generator

>>> a = np.ones((4, 5), dtype=int)

>>> a *= 2

>>> a
array([[2, 2, 2, 2, 2],
       [2, 2, 2, 2, 2],
       [2, 2, 2, 2, 2],
       [2, 2, 2, 2, 2]])

>>> b = rg.random((4, 5))

>>> b += a

>>> b
array([[2.51182162, 2.9504637 , 2.14415961, 2.94864945, 2.31183145],
       [2.42332645, 2.82770259, 2.40919914, 2.54959369, 2.02755911],
       [2.75351311, 2.53814331, 2.32973172, 2.7884287 , 2.30319483],
       [2.45349789, 2.1340417 , 2.40311299, 2.20345524, 2.26231334]])

# >>> a += b  # b is not automatically converted to integer type

~~~

Farklı türlerdeki Array'ler üzerinde çalışırken her zaman daha genele *upcasting* edildiğini unutmayın.

~~~Python

>>> import numpy as np

>>> from numpy import pi

>>> a = np.ones(5, dtype = np.int32)

>>> b = np.linspace(0, pi, 5)

>>> b.dtype.name
'float64'

>>> c = a + b

>>> c
array([1.        , 1.78539816, 2.57079633, 3.35619449, 4.14159265])

>>> c.dtype.name
'float64'

>>> d = np.exp(c * 1j)

>>> d
array([ 0.54030231+0.84147098j, -0.21295842+0.97706126j,
       -0.84147098+0.54030231j, -0.97706126-0.21295842j,
       -0.54030231-0.84147098j])

>>> d.dtype.name
'complex128'

~~~

ndarray sınıfının Array'ler üzerinde uyguladığı bir çok tekil yöntem vardır. Örneğin Array'daki tüm öğelerin toplamını hesaplamak için kullandığımız *.sum()* gibi...

~~~Python

>>> import numpy as np

>>> a = rg.random((4, 5))

>>> a
array([[0.51088888, 0.75303021, 0.14792204, 0.81962672, 0.68328691],
       [0.78709694, 0.19161626, 0.80236416, 0.19132393, 0.08155262],
       [0.85522697, 0.8612835 , 0.8765371 , 0.47190972, 0.27404839],
       [0.00709183, 0.6457209 , 0.71990938, 0.83556922, 0.28187783]])

>>> a.sum()
10.797883484611834

>>> a.min()
0.007091828603166261

>>> a.max()
0.8765370964165805

~~~

Tekil yöntemler uygulandığında varsayılan olarak Array sanki sayı listesiymiş gibi algılanır ve işlemler bu sayı listesine göre uygulanır. Peki bu işlemleri eksen belirterek (axis parameter) uygularsak nasıl olur. Bence çok güzel olur. Hadi deneyelim.

~~~Python

>>> import numpy as np

>>> b = np.arange(20).reshape(4, 5)

>>> b
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])

>>> b.sum(axis=0)     # sum of each column
array([30, 34, 38, 42, 46])

>>> b.min(axis=1)     # min of each row
array([ 0,  5, 10, 15])

>>> b.cumsum(axis=1)  # cumulative sum along each row
array([[ 0,  1,  3,  6, 10],
       [ 5, 11, 18, 26, 35],
       [10, 21, 33, 46, 60],
       [15, 31, 48, 66, 85]])

~~~

---

### <a id="universal_functions" /> Evrensel Fonksiyonlar

NumPy sin, cos, exp gibi tanıdık matematik işlevlerini bize sağlar ve bunları Evrensel Fonksiyonlar ("Universal Functions") olarak tanımlar. Bu fonksiyonlar Array'ler üzerinde element bazlı çalışırlar ve çıktı olarak da bir Array üretirler.

~~~Python

>>> import numpy as np

>>> B = np.arange(5)

>>> B
array([0, 1, 2, 3, 4])

>>> np.exp(B)
array([ 1.        ,  2.71828183,  7.3890561 , 20.08553692, 54.59815003])

>>> np.sqrt(B)
array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ])

>>> C = np.array([2., -1., 4., 5., 8.])

>>> np.add(B, C)
array([ 2.,  0.,  6.,  8., 12.])

~~~

Bakınız: all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, invert, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where

---

### <a id="indexing_slicing_and_iterating" /> İndeksleme, Dilimleme ve Yineleme

Tek boyutlu Array'ler Python'da listeler ve diziler gibi indekslenebilir dilimlenebilir ve yenileme yapılabilir..

~~~Python

>>> import numpy as np

>>> a = np.arange(10)**3

>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])

>>> a[2]
8

>>> a[2:5]
array([ 8, 27, 64])

>>> # equivalent to a[0:6:2] = 1000;

>>> # from start to position 6, exclusive, set every 2nd element to 1000

>>> a[:6:2] = 1000

>>> a
array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729])

>>> a[::-1]  # reversed a
array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000])

>>> for i in a:
        print(i**(1 / 3.))

9.999999999999998
1.0
9.999999999999998
3.0
9.999999999999998
4.999999999999999
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998

~~~

Çok boyutlu Array'ler eksen başına bir dizine sahip olabilir. Bu endeskler virgülle ayrılmış bir demet halinde olabilir.

~~~Python

impoty numpy as np

>>> def f(x, y):
        return 10 * x + y
   
>>> b = np.fromfunction(f, (5, 4), dtype=int)

>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])

>>> b[2, 3]
23

>>> b[0:5, 1]  # each row in the second column of b
array([ 1, 11, 21, 31, 41])
z
>>> b[:, 1]    # equivalent to the previous example
array([ 1, 11, 21, 31, 41])

>>> b[1:3, :]  # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])

>>> b[-1]   # the last row. Equivalent to b[-1, :]
array([40, 41, 42, 43])

~~~

b Array'inde kullanılan [] içerisinde bulundurduğu *:*'lar ile eksenleri ve öğeleri belirlemek için kullanılır. Ayrıca NumPy bu işlemleri noktalar kullanark da yapmamıza olanak sağlamaktadır. Bakınız: b[1, ...]. Noktalar tam bir indexleme olarak belirtilebilir.

Bununla birlikte Array'daki her bir eleman üzerinde bir işlem yapmak gerekirse flat özellii ile bunu yapabiliriz.

~~~Python

import numpy as np
for elemant in b.flutter

0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43

~~~

---

> Buraya kadar sabırla okuduğunuz için teşekkür eder, bu yazının güncelleneceğinden ve devamının olacağından haberdar olmanızı isterim. :-)

