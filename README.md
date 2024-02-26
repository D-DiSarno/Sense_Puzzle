# Progetto "Sense_puzzle"

## Introduzione

Benvenuti nel progetto **Sense_puzzle**, un'applicazione innovativa che sfrutta la potenza delle reti neurali Siamesi per l'embedding di sequenze genomiche. Questo progetto va oltre il semplice embedding, incorporando anche fasi cruciali come l'allineamento delle sequenze e la visualizzazione dei risultati.

Per ulteriori dettagli tecnici e informazioni approfondite, si consiglia la lettura del nostro **paper associato**. Il paper fornisce una panoramica completa delle metodologie implementate, dei risultati ottenuti e delle considerazioni teoriche alla base del progetto.

## Contenuti del Progetto
Il progetto comprende i seguenti componenti principali:

1. **siamese1.ipynb**: Questo notebook dettaglia l'implementazione del modello Siamese utilizzando PyTorch. Include anche elementi chiave e dettagli presenti nel **paper** associato al progetto.

2. **select_training_data**: Questa directory ospita l'implementazione in C++ dell'algoritmo di selezione attiva dei landmark, fondamentale per la preparazione dei dati di addestramento.

3. **DNA_Align**: Contiene l'eseguibile per valutare i risultati dell'embedding attraverso l'allineamento delle sequenze.

4. **tools**: Una raccolta di utility Python che semplificano le operazioni di manipolazione dei dati e preparazione per l'addestramento.

5. **demo**: Questa directory contiene dati di esempio utilizzati per dimostrare il funzionamento del progetto.

6. **Qiita**: Nella presente cartella è possibile accedere ai dati in formato FASTA del dataset Qiita.

8. **Risultati_iperparametri**: Contiene alcuni risultati ottenuti durante la sperimentazione del modello su diverse combinazioni di iperparametri.

9. **siamese.py**: File eseguibile con l'algoritmo genetico aggiornato.

10. **deliverables**: All'interno di questa cartella è possibile visionare la relazione relativa allo studio,la presentazione del progetto e il file di log discusso nella relazione.

## Requisiti

Prima di iniziare, assicuratevi di avere installati i seguenti componenti:

- **Clang**
- **Cmake**
- **Boost**
- **PyTorch**
- **CUDA** (opzionale ma consigliato per l'accelerazione GPU)

Per informazioni dettagliate su come installare questi strumenti, consultate le rispettive documentazioni ufficiali.

## Compilazione

Per compilare i componenti **/DNA_Align/** e **/select_training_data/**, seguite le istruzioni seguenti:

```python
cd DNA_Align
mkdir build && cd build
cmake .. && make
```
L'eseguibile si troverà in DNA_Align/build/src/.
```python
cd select_training_data
mkdir build && cd build
cmake .. && make
```
L'eseguibile sarà situato in select_training_data/build/src/.
## Esecuzione
L'esecuzione è guidata attraverso il codice Python nel file *main.py*.
- Crea un set di dati di valutazione da un file FASTA di input (seqs_0.fa).
Produce coppie di sequenze e le salva in eval.fa.
Creazione di Coppie di Sequenze:

- Utilizza uno script Python (pair.py) per generare coppie da eval.fa.
Il risultato è salvato in eval_pair.fa. 
- Allineamento delle Sequenze:
Esegue l'allineamento globale delle coppie di sequenze usando l'eseguibile C++ (nw) dalla directory DNA_Align.
Salva i risultati allineati in eval_aligned.fa.
Calcolo delle Distanze:

- Calcola le distanze tra le sequenze allineate e salva i risultati in un file di testo (eval_dist.txt).
- Selezione Attiva dei Landmark e Shuffling dei Dati di Addestramento:
Utilizza l'eseguibile C++ (select_training_data) per eseguire la selezione attiva dei landmark.
Salva le coppie selezionate e le distanze corrispondenti nei file pair.fa e dist.txt.
Applica uno shuffling ai dati di addestramento.

Avvio da Linea di Comando:
Per avviare l'intero processo da linea di comando, esegui il seguente comando:

```python
python3 main.py
```
Assicurati di trovarti nella directory principale del progetto e che tutti i prerequisiti siano stati soddisfatti prima di eseguire questo comando.

## Pulizia dataset
Per poter avere il dataset uniforme è possibile utilizzare il file *clean.py*.
Di seguito sono riportate le principali funzionalità:

- fix_fasta_format:
Questa funzione aggiorna il formato delle intestazioni nei file FASTA, introducendo il carattere "#" tra il primo e l'ultimo elemento dell'intestazione. Ciò contribuisce a una rappresentazione più coerente delle sequenze genomiche.
- remove_short_and_sequences_with_N:
Rimuove sequenze con lunghezza inferiore a 15 nucleotidi e sequenze contenenti il carattere "N". Questa operazione è essenziale per eliminare dati di bassa qualità o informazioni irrilevanti.
-count_nucleotides:
Calcola il conteggio totale delle sequenze nucleotidiche presenti in un file FASTA. Questo parametro fornisce una panoramica del volume dei dati genomici.
## Concatenazione dataset
Il file *concat.py* permette di unire diversi file in formato FASTA.
## Notebook "siamese1.ipynb"
Il notebook siamese1.ipynb contiene l'implementazione del modello Siamese e include elementi e dettagli presenti nel paper "Siamese_puzzle".
## siamese.py
Il file prova_siamese.py permette l'esecuzione diretta del codice senza l'esecuzione di ogni cella, contiene la versione più aggiornata del codice.

## Autori
[Mattia d'Argenio](https://github.com/mattiadarg), [Davide Di Sarno](https://github.com/D-DiSarno)
Contatti: M.dargenio5@studenti.unisa.it, d.disarno3@studenti.unisa.it
## Contributi e Licenza
Il progetto è open source e accetta contributi.
Per ulteriori dettagli sull'embedding, l'allineamento e la visualizzazione dei risultati, consultare il paper associato a questo progetto.