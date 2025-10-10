Proyecto: Desafío RL - A2C / PPO / SAC sobre PointEnv (propio)

Contenido:
- a2c_pointenv.py   : A2C en entorno discreto {-1,0,+1}
- ppo_pointenv.py   : PPO (clipped surrogate) en entorno discreto
- sac_pointenv.py   : SAC en entorno continuo (acción en [-1,1])
- output_*          : carpetas generadas con modelos, plots y PDFs

Requisitos:
- Python 3.8+
- PyTorch (pip install torch) - si usás GPU, instala versión con CUDA
- numpy, matplotlib

Cómo ejecutar:
- python a2c_pointenv.py
- python ppo_pointenv.py
- python sac_pointenv.py

Qué generan:
- Modelo: .pth (o bundle)
- convergence_*.png: gráfica Reward vs episodio (con suavizado exponencial)
- *_PointEnv_Report.pdf: PDF de 3 páginas con descripción, fragmentos de código y gráfica
- README.txt dentro del output con rutas y resumen

Notas:
- A2C y PPO usan el mismo entorno discreto; SAC usa entorno continuo.
- Si querés usar SAC con entorno discreto tendríamos que adaptar la versión de SAC para acciones discretas (cambio no trivial).
- Para mejorar convergencia:
  - usar más episodios (1000+)
  - normalizar observaciones
  - usar GAE (PPO ya lo hace)
  - experimentar lr, coeficientes y tamaño de red
- Para reproducibilidad fijar seeds y controlar nondeterminism en PyTorch si es crítico.

Sugerencias de entrega PDF:
- Cada script ya genera un PDF de 3 páginas. Podés combinar o editar el contenido si querés más explicaciones.
