



def funcion_binomial(k,n,p,grafico='desactivado'):
  '''
  k = casos de exito
  
  n = muestra representativa tomada
  
  p = probabilidad de exito


  1. El experimento consiste en una serie de n ensayos idénticos.
  2. En cada ensayo hay dos resultados posibles. 
  A uno de estos resultados se le llama éxito y al otro se le llama fracaso.
  3. La probabilidad de éxito, que se denota p, no cambia de un ensayo a otro.
  Por ende, la probabilidad de fracaso, que se denota 1 - p, tampoco cambia de un ensayo a otro.

  4. Los ensayos son independientes.

  5.Binario /booleano /si-No, 
  la decision de uno no condicsiona la del otro.

  ej; #Probabilidad de obtener 3 caras en 5 lanzamientos y una probabilidad de éxito del 0.5.
  print(funcion_binomial(3,5,0.5)) 
  0.3125

  '''
  from math import factorial
  import matplotlib.pyplot as plt
  from scipy import stats 
  import seaborn as sns
  import numpy as np
  
  num_exitos = factorial(n) #Factorial de la cantidad de casos de éxito buscados.
  num_eventos = factorial (k) * factorial(n-k) #Factorial del espacio muestral.
  exitos_fracaso = pow(p,k) * pow(1-p,(n-k)) # Probabilidad de exitos y fracasos.

  binomial = (num_exitos / num_eventos) * exitos_fracaso #Aplicación de la función binomial.

  if grafico == 'activado':
    N = n 
    p = p #parametros de forma 
    binomial_graf = stats.binom(N, p) # Distribución
    x = np.arange(binomial_graf.ppf(0.01),
                  binomial_graf.ppf(0.99))
    fmp = binomial_graf.pmf(x) # Función de Masa de Probabilidad
    plt.plot(x, fmp, '--') #Esta función recibe un conjunto de valores x e y y los muestra en el plano como puntos unidos por línea.
    plt.vlines(x, 0, fmp, colors='b', lw=10, alpha=0.5) #Esta función da formato a las figuras.
    plt.title('Distribución Binomial') #Esta función asigna un título.
    plt.ylabel('probabilidad') #Esta función etiqueta el eje Y.
    plt.xlabel('valores') #Esta función etiqueta el eje X.
    plt.show() #Esta función muestra las figuras

  return binomial


def probabilidad_poisson(lamba_np,x,grafico='desactivado'):
  '''
  
  lamba_np = En promedio, en el lapso de tiempo que me pide la probabilidad cuantos sucesos ocurren. funcion normalizacion.
  
  x = Casos que cumplen la condicion
  

  Se suele usar para estimar el número de veces que sucede un hecho determinado (ocurrencias)
  en un intervalo de tiempo o de espacio.

  
  Cantidad de veces que ocurre algo en un intervalo determinado (finito) 
  de x (por ej, de tiempo). 
  ¿Cuanto ocurre de ese suceso en ese intervalo de tiempo? en promedio.


  ej:
  en ese periodo entran... quiero saber la probabilidad de que entren ... en un lapso de tiempo de:

  A la oficina de reservaciones de una aerolínea llegan 48 llamadas por hora.
  Calcule la probabilidad de recibir cinco llamadas en un lapso de 5 minutos.

  print(probabilidad_poisson((48/60*5),5)) 
  lambda = primer argumento lo debo convertir a misma unidad: 48 en 60 min... en 5' recibo = 5 * 48 / 60 =4 
  
  48/60*5 = 4
  '''

  from math import e,factorial
  import matplotlib.pyplot as plt
  from scipy import stats 
  import seaborn as sns
  import numpy as np
  
  probabilidad = (pow(e,-lamba_np) * pow(lamba_np,x))/factorial(x)

  if grafico == 'activado': 
    mu =  lamba_np # parametro de forma 
    poisson = stats.poisson(mu) # Distribución
    x = np.arange(poisson.ppf(0.01),
                 poisson.ppf(0.99))
    fmp = poisson.pmf(x) # Función de Masa de Probabilidad
    plt.plot(x, fmp, '--')
    plt.vlines(x, 0, fmp, colors='green', lw=5, alpha=0.5)
    plt.title('Distribución Poisson')
    plt.ylabel('probabilidad')
    plt.xlabel('valores')
    plt.show()

    # # histograma
    # aleatorios = poisson.rvs(1000)  # genera aleatorios
    # cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
    # plt.ylabel('frequencia')
    # plt.xlabel('valores')
    # plt.title('Histograma Poisson')
    # plt.show()

  return probabilidad


def probabilidad_hipergeometrica(N,X,n,x,grafico='desactivado'):
  '''
  N = Población.

  X = totalidad de eventos en toda la poblacion (N) que si cumplen la condición.
  
  n = Muestra seleccionada.
  
  x = Cantidad de eventos en la Muestra seleccionada (n) que cumplirían la condicion.
    
  Si en un grupo de X = (población), si tengo X = (los eventos que cumplen con esa condicion en la población)
  ¿Cual es la probabilidad de que en una muestra = (n) tomada, haya x = (cantidad de eventos que cumplen con la condición en la muestra = (n))

  P = rta.

  También se responde x si o x no, pero la ocurrencia del primer evento afecta la del segundo.
  Los ensayos no son independientes y la probabilidad de éxito varía de ensayo a ensayo. NO REPO!
  
  En un ensayo de hipergeométricas, el resultado de una observación es afectado por los resultados de las observaciones previas,
  por tanto las probabilidades son condicionales.


  '''
  from math import factorial
  import matplotlib.pyplot as plt
  from scipy import stats 
  import seaborn as sns
  import numpy as np

  Xx = factorial(X)/(factorial(x)*factorial(X-x))
  NX_nx= factorial(N-X)/(factorial(n-x)*factorial((N-X)-(n-x)))
  Nn = factorial(N)/(factorial(n)*factorial(N-n))
  hipergeometrica = (Xx * NX_nx)/Nn


  if grafico == 'activado':
    # Graficando Hipergeométrica
    M, n, N = N, n, X # parametros de forma 
    hipergeometrica_graf = stats.hypergeom(M, n, N) # Distribución
    x = np.arange(0, n+1)
    fmp = hipergeometrica_graf.pmf(x) # Función de Masa de Probabilidad
    plt.plot(x, fmp, '--')
    plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
    plt.title('Distribución Hipergeométrica')
    plt.ylabel('probabilidad')
    plt.xlabel('valores')
    plt.show()

  return hipergeometrica




def Estandarizar_a_dist_Normal(x,mu,sigma,grafico='desactivado'):
  '''
  Para adecuar, estandarizar los valores de mi muestra a la funcion normal.
  
  x = Valores que cumplen la condicion dentro de la muestra seleccionada ( x = Variable aleatoria).

  mu = media

  sigma = Desviación Standard

  z = función Normal estandarizada 



  z = ( x - mu ) / sigma = VALOR A BUSCAR EN LA TABLA
    
  Las distribuciones normales son calculadas mediante la distribución normal estándar. 

  Esto es, cuando distribución normal con una media μ cualquiera y una desviación estándar (sigma) 
  cualquiera, las preguntas sobre las probabilidades en esta distribución se responden pasando primero a la distribución normal estándar.
    
  Usa las TABLAS de probabilidad normal estándar y los valores apropiados de z para hallar las probabilidades deseadas.
  
  ej:
  La calificación promedio de los estudiantes de Henry es de 78 con una desviación estandar de 25. 
  Cual es la probabilidad de tener estudiantes con calificaciones mayores o iguales 90.

  x = 90
  
  mu = 78
  
  sigma = 25 # desvio std
  
  z = float() y me dice a donde ir a buscar mi valor en el grafico de la distribucion (TABLA).

  En el ejemplo z = 0.48
  buscando z en la tabla de distribución normal obtengo que la probabilidad del ejemplo es = 0.3156
    
  '''
  
  import matplotlib.pyplot as plt
  from scipy import stats 
  import numpy as np

  z = ( x - mu ) / sigma 

  if grafico=='activado':
      # Graficando Normal
      normal = stats.norm(mu, sigma)
      x = np.linspace(normal.ppf(0.01),
                      normal.ppf(0.99), 100)
      fp = normal.pdf(x) # Función de Probabilidad
      plt.plot(x, fp)
      plt.title('Distribución Normal')
      plt.ylabel('probabilidad')
      plt.xlabel('valores')
      plt.show()

  return z # nomralizacion atar los datos de la muesta al modelo de distribucion normal. valor a buscar en tabla!

