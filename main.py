import os
import subprocess

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione del comando: {command}")
        print(f"Output dello stderr:\n{e.stderr.decode()}")
        exit(1)

def main():

    # Comandi per la preparazione dell'evaluazione
    run_command("python3 tools/sample.py -i converted/seqs_0.fa -o converted/eval.fa -s 0 -n 500")
    run_command("python3 tools/pair.py -i converted/eval.fa -o converted/eval_pair.fa")
    run_command("./DNA_Align/build/src/nw converted/eval_pair.fa converted/eval_aligned.fa")
    run_command("python3 tools/dist.py -i converted/eval_aligned.fa -o converted/eval_dist.txt")
    # Comandi per la selezione e lo shuffling dei dati di addestramento
    run_command("./select_training_data/build/src/select_training_data -f converted/seqs_0.fa -s converted/seqs_ids.txt -p converted/pair.fa -d converted/dist.txt -a 1 -t 20 -n 500")
    run_command("python3 tools/shuffle.py -p converted/pair.fa -d converted/dist.txt -s 0")

if __name__ == "__main__":
    main()