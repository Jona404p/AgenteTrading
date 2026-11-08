# Closed-Loop Trading Agent (Binance) — Fases 1 y 2

Estado: Fases 1–4 implementadas; Fase 5 disponible en modo paper con simulación
local de fills y panel web. La ejecución de órdenes reales sigue sin implementarse.

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # instala solo el langchain-* de tu proveedor LLM
cp .env.example .env                                   # rellena keys del Testnet y del LLM
```

## Comandos
```bash
pytest -v                            # 16 tests, sin red ni LLM real
python scripts/check_connection.py   # precio (+ balance si hay keys) en el entorno de .env
python -m scripts.download_data      # histórico -> data/raw y features -> data/processed
python scripts/run_stream.py         # order book y velas en vivo (Ctrl+C para salir)
python scripts/run_sentiment.py      # una pasada de sentimiento por activo
python scripts/run_sentiment.py --loop
python scripts/run_user_stream.py   # fills privados y cartera conciliada (solo lectura)
python scripts/dashboard.py        # solo panel, aunque todavía no haya modelo entrenado
python main.py --paper --dashboard --symbols BTC/USDT  # tras entrenar el modelo BTC
```

Para aumentar la historia usada al entrenar (por ejemplo, solicitar hasta cinco
años disponibles en el exchange), vuelve a descargar el rango completo:

```bash
python -m scripts.download_data --refresh --history-days 1825
python -m scripts.train_rl --symbol BTC/USDT
```

La descarga conserva y fusiona el histórico existente, elimina timestamps
duplicados y descarta la vela abierta. El entrenamiento ahora rechaza series
desordenadas, duplicadas, incompletas o con valores OHLCV/features no finitos;
el reporte del modelo registra el intervalo y tamaño de cada split temporal.
También guarda una referencia normalizada solo desde train para PSI en vivo.

El loop de `main.py` requiere un modelo entrenado para cada símbolo habilitado
en `data/models/` y acceso de red a los endpoints públicos para histórico y
WebSocket. Por ejemplo, para entrenar un modelo BTC con los datos existentes:

```bash
python -m scripts.train_rl --symbol BTC/USDT
python main.py --paper --dashboard --symbols BTC/USDT
```

Para revisar y autorizar cada cambio de posición desde el panel, inicia PAPER con:

```bash
python main.py --paper --dashboard --manual-approval --symbols BTC/USDT
```

El modelo calcula la señal al cierre de cada vela y el gestor de riesgo la filtra,
pero no se simula ninguna orden hasta pulsar **Aprobar operación PAPER**. El panel
muestra lado, cantidad y notional estimados, exposición actual/objetivo, motivos
de riesgo y hora de caducidad. Las propuestas caducan en dos minutos o al procesarse
una vela nueva; antes de ejecutar se vuelve a evaluar el riesgo y, si cambia el
objetivo aprobado, se cancela la propuesta. Rechazarla no modifica la cartera.
Este control es exclusivamente PAPER; no habilita órdenes reales. En Spot, las
señales negativas del modelo se interpretan como exposición cero, igual que en el
entorno de entrenamiento; no se convierten en órdenes cortas.

El entrenamiento usa `RL_TOTAL_TIMESTEPS` (300 000 por defecto) y puede tardar.
Para añadir ETH, primero descarga sus datos y entrena también su propio modelo.
El modelo `SYNTH/USDT` creado con `--synthetic` sirve únicamente para probar el
pipeline; no sustituye un modelo entrenado con el histórico del activo. En esta
copia, `main.py` solo permite `--paper`; no hay gateway para enviar órdenes reales.

## Panel web (opcional)

El proyecto incluye un panel web para observar el estado, las operaciones, el
riesgo, el drift, el sentimiento y los logs del bot; también puede escribir el
archivo de control `HALT`. Su servidor está en `src/dashboard` y escucha en
`127.0.0.1:8765` por defecto. Para integrarlo con el bot de Fase 5, configura
`DASHBOARD_HOST`, `DASHBOARD_PORT` y, si se enlaza fuera de loopback, un
`DASHBOARD_TOKEN` fuerte. No expongas el panel públicamente sin autenticación y
controles de red adecuados.

El panel puede iniciarse por separado antes de tener modelos. Cuando se ejecuta
con `main.py --paper --dashboard`, el orquestador escribe el estado y vigila el
archivo `data/live/HALT`; al activarlo, las posiciones simuladas se aplanan y el
loop se detiene. El control afecta solo a la simulación local, no a Binance.
Las tarjetas de drift y sentimiento muestran su estado (deshabilitado,
esperando datos, calentamiento o resultado) en vez de `null`; el estado de
sentimiento se persiste en `status.json`. Drift CRITICAL bloquea aperturas y
aumentos de exposición, pero permite reducir o cerrar.

## User Data Stream y conciliación

`run_user_stream.py` escucha el User Data Stream autenticado de Binance en Spot
(`userDataStream.subscribe.signature` por la WebSocket API, eventos `executionReport`)
o USDⓈ-M Futures (`ORDER_TRADE_UPDATE`), registra los cambios
de estado/rechazos en `data/execution/orders_<mercado>.jsonl` y cada fill parcial
con precio y comisión exactos en `data/execution/fills_<mercado>.jsonl`. También
reconcilia periódicamente `PortfolioState` desde balance/posiciones y precios
obtenidos con CCXT. RiskManager exige por defecto snapshots con `source=exchange`;
la excepción `require_exchange_state=False` es solo para simulaciones y pruebas.

En Spot Testnet el stream usa `wss://ws-api.testnet.binance.vision/ws-api/v3`;
no necesita el endpoint REST antiguo de `listenKey`. Si la cuenta contiene un
activo sin mercado contra la divisa quote (por ejemplo, un token de prueba no
negociable), la conciliación rechaza todo el snapshot: no estima ese saldo a cero
ni crea un equity parcial. Usa una cuenta Testnet que solo tenga activos
valorables o espera al reset de Testnet; no desactives esta validación para operar.

El proceso **no envía ni cancela órdenes**. Requiere credenciales del entorno
seleccionado con permiso de lectura, sin retiros; configura esos permisos y la
lista blanca de IP manualmente en Binance. Los archivos locales guardan información
financiera privada y están excluidos del repositorio mediante `.gitignore`.

## Seguridad
- `BINANCE_ENV=testnet` por defecto. `live` exige además `CONFIRM_LIVE_TRADING=YES`.
- Topes en código, independientes de la configuración: pérdida diaria máxima de 3%,
  apalancamiento total máximo de 1×, cambio máximo de exposición de 25% del equity
  por orden y stop-loss de hasta 2 ATR. Los cierres forzados por stop o freno diario
  y las correcciones para respetar el apalancamiento máximo pueden superar el tamaño
  normal por orden para permitir liquidar o volver a un nivel de exposición permitido.
- Al superar la pérdida diaria, el gestor de riesgo persiste `DAILY_HALT`, emite la
  decisión `FLATTEN` y bloquea nuevas entradas hasta el siguiente día UTC. Si el
  estado persistido del freno está dañado, inicia bloqueado (`DRAWDOWN_HALT`).
- El ejecutor de `main.py --paper` solo simula fills localmente. No existe un
  gateway de órdenes reales conectado al gestor de riesgo; `FLATTEN` no cancela
  órdenes ni cierra posiciones en Binance. No operes dinero real suponiendo que
  los stops o el freno liquidarán automáticamente.
- No subas `.env` al repositorio. En el panel de Binance desactiva explícitamente
  retiros para las API keys, habilita únicamente los permisos necesarios y restringe
  las keys a IP fijas permitidas. Usa keys separadas para Testnet y producción.
- El software no puede aplicar ni verificar los permisos de Binance o la lista
  blanca de IP; configúralos directamente en Binance y no compartas las keys.
- El histórico de entrenamiento se baja de endpoints públicos de producción (`HISTORICAL_FROM_LIVE=true`);
  el Testnet se usa solo para ejecución.

## Notas
- Futures: cableado pero sin validar (el testnet de futuros de Binance cambió; revisa `enable_demo_trading` en tu versión de ccxt).
- No generes scores LLM retroactivos para entrenar el RL (sesgo de anticipación); el histórico limpio
  se acumula en `data/sentiment/*.jsonl`.
# AgenteTrading
