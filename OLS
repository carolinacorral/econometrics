import numpy as np

class RegresionLinealMultiple:
  def __init__(self):
    self.coefs = []
    self.residuales = []
    self.r_cuadrado = 0
    self.r_cuadrado_ajustado = 0
    self.ecm = 0
    self.ee_ = []
    self.t_val = []

  def fit(self,X,y,metodo='matriz',unbiased_ecm=True):
    """
    Ajusta un modelo de regresión lineal múltiple a los datos dados.

    Args:
        X (np.ndarray): Matriz de características (n x p).
        y (np.ndarray): Vector objetivo (n,).
        metodo (str, optional): Método para estimar los coeficientes. Actualmente solo 'matriz'. Default es 'matriz'.
        unbiased_ecm (bool, optional): Si True, calcula el ECM no sesgado dividiendo entre (n - k).
                                       Si False, divide entre n. Default es True.

    Returns:
        None
    """
    if X.ndim == 1:
      X = np.array([X]).T
      unbiased_ecm = False

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # agregar coeficiente beta_0
    y = np.array(y)

    if metodo == 'matriz':
      cov = np.linalg.inv(X.T @ X)
      b = cov @ (X.T @ y)

    self.coefs = b

    y_hat = self.predict(X,add_intercept=False)

    # -- Diagnósticos --

    # Residuales
    self.residuales = y - y_hat 

    # Error cuadrático medio
    if unbiased_ecm: # insesgado
      n, k = X.shape
      self.ecm = np.sum(self.residuales**2) / (n - k)
    else:
      self.ecm = (1/len(y)) * np.sum(self.residuales**2)

    # Error estándar por coeficiente
    std_err = np.diag(cov) # Diagonal de matriz de covarianza
    self.ee_ = [ np.sqrt(x * self.ecm)  for x in std_err] # SE 

    # Valores t
    self.t_val = [b / se for b, se in zip(self.coefs, self.ee_)] 

    # R cuadrado
    y_bar = np.mean(y)
    self.r_cuadrado = np.sum((y_hat - y_bar) ** 2) /  np.sum((y - y_bar) ** 2)
    # R cuadrado (ajustado)
    self.r_cuadrado_ajustado = 1 - (1 - self.r_cuadrado) * (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)


  def predict(self,X,add_intercept=True):
    """
    Realiza predicciones usando el modelo ajustado.

    Args:
        X (np.ndarray): Matriz de características (n x p).
        add_intercept (bool, optional): Si True, agrega una columna de unos como intercepto. Default es True.

    Returns:
        np.ndarray: Vector de predicciones.
    """

    if add_intercept:
      X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    return X @ self.coefs


  def summary(self, feature_names=None):
    """
    Imprime un resumen con diagnósticos del modelo OLS ajustado:
    coeficientes, errores estándar, valores t, R² y R² ajustado.

    Args:
        feature_names (list of str, optional): Lista con los nombres de las variables independientes.
                                               Si no se proporciona, se usarán nombres genéricos como x0, x1, etc.

    Returns:
        None
    """
    if feature_names is None:
        feature_names = [f'x{i}' if i != 0 else 'Inter.' for i in range(len(self.coefs))]
    else:
        feature_names = ['Inter.'] + feature_names

    print(f"Diagnóstico de OLS\n")
    print(f"R2: {self.r_cuadrado:.3f}\tR2_ajustado: {self.r_cuadrado_ajustado:.3f}\tECM: {self.ecm:.3f}")
    print(f"\n{'Variable':<10}{'Coef.':>10}{'E.E.':>10}{'t':>10}")
    for var, coef, ee, t in zip(feature_names, self.coefs, self.ee_, self.t_val):
        print(f"{var[:8]:<10}{coef:>10.4f}{ee:>10.4f}{t:>10.4f}")
