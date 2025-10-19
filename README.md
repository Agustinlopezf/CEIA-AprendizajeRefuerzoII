<img src="https://github.com/hernancontigiani/ceia_memorias_especializacion/raw/master/Figures/logoFIUBA.jpg" width="500" align="center">


## Descripción
Este repositorio contiene la resolución del **Desafío Integrador** de la asignatura **Aprendizaje por Refuerzo II**, correspondiente a la **Carrera de Especialización en Inteligencia Artificial (CEIA)** de la **Universidad de Buenos Aires (UBA)**.

El objetivo principal es **comparar el desempeño de tres algoritmos de Deep Reinforcement Learning (RL)** —**A2C**, **PPO** y **SAC**— en un entorno continuo, analizando su capacidad de aprendizaje, estabilidad y convergencia.

El entorno utilizado es **BipedalWalker-v3** de la biblioteca **Gymnasium**, donde un agente bípedo debe aprender a caminar sobre un terreno irregular optimizando su equilibrio y desplazamiento.

---

## Autor
- **Agustín López Fredes**

---

## Estructura del Repositorio
- **Notebook principal:** `comparativa_algoritmos_bipedal.ipynb`
- **Modelos guardados:** `models/{a2c, ppo, sac}/best_model.zip`
- **Normalización:** `models/{a2c, ppo, sac}/vecnormalize.pkl` (cuando aplica)
- **Logs de entrenamiento y evaluación:** `logs/{a2c, ppo, sac}/`
  - `progress.csv` (métricas de entrenamiento)
  - `evaluations.npz` (recompensas y longitudes de episodios de evaluación)
- **TensorBoard:** `{a2c, ppo, sac}_tensorboard/`
- **Videos de políticas:** `videos/{a2c, ppo, sac}/`

---

## Algoritmos Implementados

### A2C (Advantage Actor-Critic)
- **Tipo:** on-policy, actor-crítico síncrono.  
- **Descripción:** El agente actualiza la política (actor) y la función de valor (crítico) en paralelo, usando la ventaja estimada:  
  A(s, a) = Q(s, a) - V(s)  
- **Ventajas:** simple, directo y estable.  
- **Limitaciones:** menos eficiente en entornos continuos, requiere más muestras.

### PPO (Proximal Policy Optimization)
- **Tipo:** on-policy, política con clipping.  
- **Descripción:** Mejora A2C restringiendo los cambios grandes en la política entre actualizaciones sucesivas.  
  L(CLIP) = E[ min( r_t * A_t , clip(r_t, 1 - e, 1 + e) * A_t ) ]  
- **Ventajas:** estable, robusto y ampliamente usado.  
- **Limitaciones:** requiere mayor cantidad de timesteps y ajuste fino de hiperparámetros.

### SAC (Soft Actor-Critic)
- **Tipo:** off-policy, máxima entropía.  
- **Descripción:** Optimiza la recompensa esperada junto con la entropía de la política para promover exploración.  
  J(pi) = E[ r(s,a) + alpha * H(pi(.|s)) ]  
- **Ventajas:** alta estabilidad, mejor exploración y eficiencia de datos.  
- **Limitaciones:** mayor complejidad y consumo de memoria.

---

## Características del Entorno
- **Entorno:** BipedalWalker-v3 (Gymnasium)
- **Tipo:** continuo  
- **Acciones:** vector de 4 dimensiones (fuerzas aplicadas en articulaciones)
- **Observaciones:** vector de 24 variables (posición, velocidad, ángulos, sensores, etc.)
- **Recompensa:** positiva por avanzar, penalización por caídas o movimientos ineficientes
- **Objetivo:** recorrer la mayor distancia posible sin caer

---

## Metodología
- **Implementación:** Stable Baselines3 sobre Gymnasium y PyTorch  
- **Entrenamiento:** 3.000.000 de timesteps por algoritmo  
- **Evaluación periódica:** `EvalCallback` con guardado del mejor modelo y métricas (`evaluations.npz`)  
- **Normalización:** `VecNormalize` para observaciones y recompensas  
- **Registro:** métricas en `progress.csv` y TensorBoard  

---

## Mejoras y Ajustes Principales
- **Aumento de timesteps:** se extendió el entrenamiento a 3.000.000 pasos para garantizar convergencia.  
- **Normalización:** uso de `VecNormalize` para estabilizar el aprendizaje.  
- **Arquitectura:** redes MLP de dos capas de 256 unidades.  
- **Learning rate:**  
  - A2C y PPO: 3e-5  
  - SAC: 3e-4 con coeficiente de entropía automático.  
- **Parámetros SAC:**  
  - Buffer size: 500.000  
  - Batch size: 256  
  - Train frequency: 64  
  - Gradient steps: 64  
  - Tau: 0.02  

---

## Resultados y Gráficos
- **Curvas de convergencia:** recompensas medias por timestep con desviación estándar.  
- **Comparación final:** recompensa promedio final por algoritmo.  
- **Videos:** ejecución de los tres agentes entrenados en `videos/{a2c, ppo, sac}/`.  

> Los gráficos se generan automáticamente a partir de los archivos `evaluations.npz`.

---

## Conclusiones
- **SAC** mostró el mejor desempeño global y mayor estabilidad.  
- **PPO** presentó convergencia estable y desempeño alto.  
- **A2C** sirvió como baseline on-policy, con aprendizaje más lento.  

**Conclusión general:**  
El algoritmo **SAC** resultó el más eficiente en entornos continuos como *BipedalWalker-v3*, seguido de **PPO** por su equilibrio entre estabilidad y rendimiento.

---

## Requisitos y Ejecución
- **Python:** 3.12  
- **Instalación con Poetry:**  
  ```bash
  poetry install
  ```
- **Ejecución:**  
  Abrir y ejecutar el notebook `comparativa_algoritmos_bipedal.ipynb`.  
- **Control de reentrenamiento:**  
  Cambiar la variable `reentrenar = True` o `False` según necesidad.  
- **Visualización de métricas:**  
  ```bash
  tensorboard --logdir ./ppo_tensorboard/
  ```

---

## Tecnologías Utilizadas
- **Gymnasium 1.2.1**  
- **Stable Baselines3 2.3.0**  
- **PyTorch 2.x**  
- **NumPy / Pandas / Matplotlib**  
- **Poetry**  
- **TensorBoard**

---

## Referencias
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.  
- Stable Baselines3 Documentation: https://stable-baselines3.readthedocs.io  
- Gymnasium Documentation: https://gymnasium.farama.org  
