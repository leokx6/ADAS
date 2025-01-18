# From Cones Detection to Road Segmentation

## Descrizione
Questo progetto utilizza diversi pacchetti Python per gestire modelli YOLO, elaborare immagini e video, e applicare tecniche di analisi e disegno su immagini. Di seguito trovi le istruzioni per configurare l'ambiente necessario.

## Requisiti di Sistema
Assicurati di avere installato:

- Python 3.8 o superiore
- pip (gestore di pacchetti Python)

## Installazione dei Requisiti

Per eseguire questo progetto, installa le seguenti dipendenze Python elencate in `requirements.txt` utilizzando il comando:

```bash
pip install -r requirements.txt
```

### Contenuto del file `requirements.txt`
Ecco le librerie richieste per il progetto:

```
torch
ultralytics
Pillow
opencv-python
numpy
```

### Descrizione dei Moduli

- **torch**: Libreria per il calcolo scientifico e il deep learning.
- **ultralytics**: Fornisce supporto per i modelli YOLO di ultima generazione.
- **Pillow**: Utilizzata per l'elaborazione di immagini (modifica, apertura, salvataggio).
- **opencv-python**: Utilizzata per l'elaborazione di immagini e video.
- **numpy**: Libreria per l'elaborazione numerica e il calcolo matematico.

## Struttura del Progetto
La struttura del progetto è organizzata come segue:

```
.
├── convertions/             # Moduli per la gestione delle conversioni
│   └── mask_convertion.py   # Script per la conversione delle maschere
├── documentation/           # File relativi alla documentazione
│   └── doc.html             # Documentazione HTML del progetto
├── models/                  # Directory per i modelli
│   └── yolo.pt              # Modello YOLO pre-addestrato
├── README.md                # Documentazione principale del progetto
├── configmodel.yaml         # File di configurazione del modello
├── detector.py              # Script per il rilevamento
├── requirements.txt         # Elenco delle dipendenze
├── seg_model.py             # Script per il modello di segmentazione
```

### Descrizione dei File Principali

- **convertions/mask_convertion.py**: Contiene funzioni per la conversione e manipolazione delle maschere.
- **documentation/doc.html**: File HTML per la documentazione dettagliata del progetto.
- **models/yolo.pt**: Modello YOLO pre-addestrato per il rilevamento degli oggetti.
- **configmodel.yaml**: File di configurazione per specificare i parametri del modello.
- **detector.py**: Script principale per il rilevamento degli oggetti.
- **seg_model.py**: Script per la segmentazione delle immagini.

## Come Eseguire
1. Clona o scarica il repository del progetto:

   ```bash
   git clone <url-del-repository>
   cd <nome-cartella>
   ```

2. Installa le dipendenze:

   ```bash
   pip install -r requirements.txt
   ```

3. Avvia lo script principale:

   ```bash
   python detector.py
   ```

## Note
- Assicurati che il tuo ambiente supporti GPU (opzionale ma raccomandato per l'uso con modelli YOLO).
- Verifica di avere i permessi per accedere ai file nella directory `data/` e scrivere in `outputs/`.

## Supporto
Se riscontri problemi, contatta il manutentore del progetto o apri una segnalazione nel repository GitHub.
